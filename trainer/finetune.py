import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from clip.clip import tokenize
from dataset.imagenet import ImageNet, zeroshot_classifier
from metric import AverageMeter, ClassIncrementalMetric, TaskIncrementalMetric
from trainer.utils import accuracy, get_ckpt_save_path, logging, resume


class FinetuneCLIP(object):
    def __init__(self, args):
        self.args = args

        self.num_classes = args.num_classes
        if args.scenario == 'class_incremental':
            METRIC = ClassIncrementalMetric
        elif args.scenario in ['domain_incremental', 'task_incremental']:
            METRIC = TaskIncrementalMetric
        else:
            raise ValueError
        self.metric = METRIC(args)
        self.unseen_metric = METRIC(args)
        self.full_metric = METRIC(args)
        self.zero_shot_metric = AverageMeter()

    def only_evaluation(self, model, dataset, task):
        model, _, _, _ = resume(self.args, task, model)
        self.evaluation(model, dataset, task)

    def unfreeze_model(self, model):
        model.train()
        if 'visual' in self.args.method:
            print('only finetune visual backbone')
            model.freeze(text=False)

    def get_loader(self, dataset, is_train=False):
        if dataset is None:
            return None

        sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size,
            shuffle=(sampler is None),
            num_workers=self.args.workers, sampler=sampler, drop_last=(is_train and self.args.few_shot == 0))
        return train_dataloader

    def get_iterator(self, dataset, task):
        if self.args.balanced_buffer and task > 0:
            trainset = dataset.get_dataset(
                task, is_train=True, with_buffer=False)
            bufferset = dataset.get_buffer(task) if task > 0 else None
            # print('buffer:', bufferset)
        else:
            trainset = dataset.get_dataset(
                task, is_train=True, with_buffer=(self.args.buffer_size > 0))
            bufferset = None
        # print(trainset)
        if bufferset:
            buffer_loader = self.get_loader(bufferset)
        else:
            buffer_loader = None
        train_dataloader = self.get_loader(trainset, is_train=True)
        total_batches = len(train_dataloader)

        return train_dataloader, buffer_loader, total_batches

    def compute_loss(self, batch, model, **kwargs):
        buffer = kwargs.get('buffer', None)
        epoch = kwargs.get('epoch', 0)
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        images, _, texts = batch
        if buffer and epoch > 0:
            images_b, _, texts_b = buffer
            images = torch.cat([images, images_b])
            texts = torch.cat([texts, texts_b])

        images = images.to(self.args.device)
        texts = texts.to(self.args.device)
        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=self.args.device)

        logits_per_image, logits_per_text = model(images, texts)

        total_loss = (loss_img(logits_per_image, ground_truth) +
                      loss_txt(logits_per_text, ground_truth)) / 2
        return total_loss

    def update_model(self, model, optimizer, **kwargs):
        optimizer.step()

    def get_batch_size(self, batch, **kwargs):
        return batch[0].size(0)

    def cosine_scheduler(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule

    def train(self, teacher_model, model, dataset, task):
        train_dataloader, buffer_loader, total_batches = self.get_iterator(
            dataset, task)
        momentum_schedule = self.cosine_scheduler(self.args.schedule, 1, self.args.epochs, len(train_dataloader))

        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-6,
                                   weight_decay=self.args.wd)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.args.lr,
                                  weight_decay=self.args.wd)
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.2)

        else:
            raise NotImplementedError
        if not self.args.no_scheduler:
            self.lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=self.args.epochs * 10,
                cycle_mult=1.0,
                max_lr=self.args.lr,
                min_lr=0,
                warmup_steps=1
            )
        self.unfreeze_model(model)
        batch_time = AverageMeter()
        loss = AverageMeter()
        optimizer.zero_grad()

        if self.args.sanity:
            self.args.epochs=1

        for epoch in range(self.args.epochs):
            buffer_iterator = iter(buffer_loader) if buffer_loader else None
            for iiter, batch in enumerate(train_dataloader):
                batch_size = self.get_batch_size(batch)
                end = time.time()
                if buffer_iterator:
                    try:
                        batch_b = next(buffer_iterator)
                    except StopIteration:
                        buffer_iterator = iter(buffer_loader)
                        batch_b = next(buffer_iterator)
                else:
                    batch_b = None

                total_loss = self.compute_loss(
                    batch, model, buffer=batch_b, epoch=epoch)
                total_loss.backward()

                self.update_model(model, optimizer,
                                  count=batch_size, epoch=epoch, task=task)

                optimizer.zero_grad()

                batch_time.update(time.time() - end)
                loss.update(total_loss.item() / batch_size, n=batch_size)
                logging('iter', iiter + epoch * total_batches,
                        f'train_loss/{task}', loss.val, self.args)
                if iiter % self.args.print_frequency == 0:
                    print(' Epoch: [{0}/{1}], Batch: [{2}/{3}]\t'.format(epoch, self.args.epochs, iiter, total_batches),
                          f'Batch Time {batch_time.val: .3f} ({batch_time.avg: .3f})\t'
                          f'Loss {loss.val:.4f} ({loss.avg: .4f}) \t'
                          f'Estimated Task Time {batch_time.avg * total_batches * self.args.epochs / 3600: .3f} H'
                          )
                if self.args.ema:
                    with torch.no_grad():
                        m = momentum_schedule[iiter]
                        for param_q, param_k in zip(model.parameters(), teacher_model.parameters()):
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                if self.args.sanity:
                    break
            if (epoch+1) % self.args.val_frequency == 0:
                model.eval()
                ########### commenting out this########.
                # avg = self.middle_evaluation(    model, dataset, task, epoch)
                # self.unfreeze_model(model)
            if not self.args.no_scheduler:
                self.lr_scheduler.step()

        model.eval()
        print('Update Buffer....')
        dataset.update_buffer(task)

    def eva_task_t(self, t, testset, model, task, text_features, text_features_full):
        zero_shot_metric = AverageMeter()
        avg_metric = AverageMeter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_dataloader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        for (image, label, _) in tqdm(test_dataloader, desc=f"Evaluation for {t}",
                                      total=len(test_dataloader)):
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if t <= task:  # update average accuracy for current batch
                logits = 100.0 * image_features @ text_features.T
                acc = accuracy(logits, label)[0]
                avg_metric.update(acc, image.size(0))

            # update zero-shot accuracy for current batch
            logits_full = 100.0 * image_features @ text_features_full.T
            acc_full = accuracy(logits_full, label)[0]
            zero_shot_metric.update(acc_full, image.size(0))
            if self.args.sanity:
                break

        avg = avg_metric.avg if not torch.is_tensor(
            avg_metric.avg) else avg_metric.avg.item()
        unseen_avg = zero_shot_metric.avg if not torch.is_tensor(
            zero_shot_metric.avg) else zero_shot_metric.avg.item()

        return avg, unseen_avg, len(testset)

    def zero_shot_evaluation(self, model, transform):
        testset = ImageNet(transform)
        metric = AverageMeter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        zeroshot_weights = zeroshot_classifier(testset.classes, model)
        test_dataloader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        for image, label in tqdm(test_dataloader, desc=f"Evaluation for ImageNet Validation Set",
                                 total=len(test_dataloader)):
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights
            acc = accuracy(logits, label)[0]
            metric.update(acc, image.size(0))
        return metric.avg.item()

    def middle_evaluation(self, model, dataset, task, epoch,  log_name=None):
        if hasattr(dataset, 'classifier'):
            text_features_full = dataset.classifier(
                dataset.class_name_full, model)
        else:

            text_inputs_full = torch.cat(
                [tokenize(f"a photo of a {c}") for c in dataset.class_name_full]).cuda()
            with torch.no_grad():
                text_features_full = model.encode_text(text_inputs_full)
                text_features_full /= text_features_full.norm(
                    dim=1, keepdim=True)

        if task < dataset.num_tasks - 1:
            unseen_class_idx = torch.Tensor(np.concatenate(
                dataset.task_classes[task + 1:], axis=None)).to(torch.long)

            text_features = text_features_full.clone().detach()
            text_features[unseen_class_idx] = 0
        else:
            text_features = text_features_full.clone().detach()
        acc = 0.0
        n = 0.0

        for t in range(task + 1):
            validset = dataset.get_dataset(t, is_train=False)
            acct, _, nt = self.eva_task_t(
                t, validset, model, task, text_features, text_features_full)
            acc += acct * nt
            n += nt
            print(f'acc at task {t}: {acct}')

        if self.args.report_to:
            log_name = 'average accuracy' if log_name is None else log_name
            logging('epoch', epoch,
                    f'{task}/{log_name}', acc / n, self.args)
        print(f'val acc {acc / n}')
        return acc / n

    def evaluation(self, model, dataset, task, log=True):

        unseen_metric = self.unseen_metric
        avg_metric = self.metric

        if self.args.scenario == 'class_incremental':

            if hasattr(dataset, 'classifier'):
                text_features_full = dataset.classifier(
                    dataset.class_name_full, model)
            else:
                text_inputs_full = torch.cat(
                    [tokenize(f"a photo of a {c}") for c in dataset.class_name_full]).cuda()
                with torch.no_grad():
                    text_features_full = model.encode_text(text_inputs_full)
                    text_features_full /= text_features_full.norm(
                        dim=1, keepdim=True)

            if task < dataset.num_tasks - 1:
                unseen_class_idx = torch.Tensor(np.concatenate(
                    dataset.task_classes[task + 1:], axis=None)).to(torch.long)

                text_features = text_features_full.clone().detach()
                text_features[unseen_class_idx] = 0
            else:
                text_features = text_features_full.clone().detach()
        for t in range(self.args.num_tasks):
            testset = dataset.get_dataset(t, is_train=False)
            if self.args.scenario != 'class_incremental':
                class_name = dataset.get_class_name(t)
                text_inputs_full = torch.cat(
                    [tokenize(f"a photo of a {c}") for c in class_name]).cuda()
                with torch.no_grad():
                    text_features_full = model.encode_text(text_inputs_full)
                    text_features_full /= text_features_full.norm(
                        dim=1, keepdim=True)
                text_features = text_features_full

            acc, acc_full, n = self.eva_task_t(
                t, testset, model, task, text_features, text_features_full)

            # update for current task
            self.full_metric.update(task, t, acc_full, n=n)
            self.full_metric.update_metric(task, t)
            if t <= task:
                avg_metric.update(task, t, acc, n=n)
                avg_metric.update_metric(task, t)
            else:
                unseen_metric.update(task, t, acc_full, n=n)
                unseen_metric.update_metric(task, t)
            if self.args.report_to:
                logging('task', task, f'{t}/accuracy per task', acc, self.args)

        # zero_shot = self.zero_shot_evaluation(model, dataset.transform) if not (
        #     self.args.debug or self.args.sweep) else 0
        # self.zero_shot_metric.update(zero_shot)

        if not log:
            return avg_metric.average_accuracy[task], unseen_metric.average_accuracy[task]

        print(
            f' * End evaluation: task accuracy top1 {self.metric.average_accuracy[task]:.2f}')
        print(
            f' * End evaluation: forgetting top1 {self.metric.forgetting[task]:.2f}')
        print(
            f' * End evaluation: learning top1 {self.metric.learning[task]:.2f}')
        print(
            f' * End evaluation: average learning top1 {self.metric.learning[:task+1].mean():.2f}')
        print(
            f' * End evaluation: unseen accuracy top1 {self.unseen_metric.average_accuracy[task]:.2f}')
        print(
            f' * End evaluation: whole set evaluation top1 {self.full_metric.average_accuracy[task]:.2f}')
        # print(f'* End evaluation: ImageNet zero0shto top1 {zero_shot:.2f}')

        if self.args.report_to:
            logging('task', task, 'average accuracy',
                    self.metric.average_accuracy[task], self.args)
            logging('task', task, 'forgetting',
                    self.metric.forgetting[task], self.args)
            logging('task', task, 'learning',
                    self.metric.learning[task], self.args)
            logging('task', task, 'average learning',
                    self.metric.learning[:task+1].mean(), self.args)
            logging('task', task, 'unseen accuracy',
                    self.unseen_metric.average_accuracy[task], self.args)
            # logging('task', task, 'ImageNet zero-shot accuracy',
            #         zero_shot, self.args)
            logging('task', task, 'full set accuracy',
                    self.full_metric.average_accuracy[task], self.args)
            # if task == 2:
            #     wandb.log(
            #         {'valid accuracy': self.metric.average_accuracy[task]})

    def save_checkpoint(self, model, task, args):
        if args.save_ckpt:
            path = get_ckpt_save_path(args, task)
            os.makedirs(os.path.join(args.save_base_path,
                        self.args.name), exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), }, path)

    def tta(self, teacher_model, model, dataset, task):
        if self.args.scenario == 'class_incremental':
            if hasattr(dataset, 'classifier'):
                text_features_full = dataset.classifier(dataset.class_name_full, model)
                if self.args.tta_loss_mode == "teacher_student":
                    text_features_teacher_full = dataset.classifier(dataset.class_name_full, teacher_model)

            else:
                text_inputs_full = torch.cat([tokenize(f"a photo of a {c}") for c in dataset.class_name_full]).cuda()
                with torch.no_grad():
                    text_features_full = model.encode_text(text_inputs_full)
                    text_features_full /= text_features_full.norm(dim=1, keepdim=True)

            if task < dataset.num_tasks - 1:
                unseen_class_idx = torch.Tensor(np.concatenate(dataset.task_classes[task + 1:], axis=None)).to(torch.long)
                text_features = text_features_full.clone().detach()
                text_features[unseen_class_idx] = 0
                if self.args.tta_loss_mode == "teacher_student":
                    text_features_teacher = text_features_teacher_full.clone().detach()
                    text_features_teacher[unseen_class_idx] = 0
            else:
                text_features = text_features_full.clone().detach()
                if self.args.tta_loss_mode == "teacher_student":
                    text_features_teacher = text_features_teacher_full.clone().detach()

        for t in range(self.args.num_tasks):
            if t>task:
                break # TTA will happen only for tasks seen till now.
            testset = dataset.get_dataset(t, is_train=False, is_tta=True)
            if self.args.scenario != 'class_incremental':
                class_name = dataset.get_class_name(t)
                text_inputs_full = torch.cat(
                    [tokenize(f"a photo of a {c}") for c in class_name]).cuda()
                with torch.no_grad():
                    text_features_full = model.encode_text(text_inputs_full)
                    text_features_full /= text_features_full.norm(
                        dim=1, keepdim=True)
                text_features = text_features_full
            for e in range(1):
                # Training for 5 epochs in TTA
                if self.args.tta_loss_mode == "teacher_student":
                    self.compute_tta_teacher_student_loss(t, testset, teacher_model, model, task, text_features, text_features_teacher)
                elif self.args.tta_loss_mode == "base":
                    self.compute_tta_loss(t, testset, teacher_model, model, task, text_features)
                else:
                    raise Exception("TTA loss not implemented",  self.args.tta_loss_mode)



    def compute_tta_teacher_student_loss(self,t, testset, teacher_model, model, task, text_features, text_features_teacher):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=self.args.wd)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2)

        else:
            raise NotImplementedError

        self.unfreeze_model(model)
        # batch_time = AverageMeter()
        loss = AverageMeter()
        optimizer.zero_grad()
        test_dataloader = DataLoader(testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        momentum_schedule = self.cosine_scheduler(self.args.schedule, 1, 1, len(test_dataloader))

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        for iiter, (image, label, _) in tqdm(enumerate(test_dataloader), desc=f"TTA for {t}", total=len(test_dataloader)):
            image = image.to(device)
            label = label.to(device)
            ground_truth_batch = torch.arange(len(image), dtype=torch.long, device=self.args.device) 
            with torch.no_grad():
                image_features = model.encode_image(image) # Obtain image features from student
                teacher_image_features = teacher_model.encode_image(image) # # Obtain image features from Teacher
            image_features /= image_features.norm(dim=-1, keepdim=True)
            teacher_image_features /= teacher_image_features.norm(dim=-1, keepdim=True)
            if t <= task:  # update average accuracy for current batch
                if self.args.oracle: 
                    predicted_class = label 
                else:
                    logits_per_image_teacher = teacher_model.logit_scale.exp() * teacher_image_features @ text_features_teacher.T
                    logits_per_image_student = model.logit_scale.exp() * image_features @ text_features.T

                    predicted_class_teacher = torch.argmax(logits_per_image_teacher, dim=1)
                    predicted_class_student = torch.argmax(logits_per_image_student, dim=1)
                    predicted_class = torch.max(predicted_class_student, predicted_class_teacher)

                text_features_predicted = text_features[predicted_class]
                logits_per_image_phase_2 = torch.mul(model.logit_scale.exp() * image_features @ text_features_predicted.T, 100)
                logits_per_text_phase_2 = logits_per_image_phase_2.T
            
                total_loss = (loss_img(logits_per_image_phase_2, ground_truth_batch) + loss_txt(logits_per_text_phase_2, ground_truth_batch)) / 2.0
                total_loss.backward()
                loss.update(total_loss.item() / image.shape[0], n=image.shape[0])
                self.update_model(model, optimizer, count=image.shape[0], task=t)
                optimizer.zero_grad()
                with torch.no_grad(): # Updating the Teacher via EMA 
                    m = momentum_schedule[iiter]
                    for param_q, param_k in zip(model.parameters(), teacher_model.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                if iiter % self.args.print_frequency == 0:
                    print(f'TTA Loss {loss.val:.4f} ({loss.avg: .4f}) \t')

    def compute_tta_loss(self, t, testset, teacher_model, model, task, text_features):
        device = "cuda" if torch.cuda.is_available() else "cpu"
       
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-6,
                                   weight_decay=self.args.wd)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.args.lr,
                                  weight_decay=self.args.wd)
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.2)

        else:
            raise NotImplementedError

        self.unfreeze_model(model)
        # batch_time = AverageMeter()
        loss = AverageMeter()
        optimizer.zero_grad()
        test_dataloader = DataLoader(testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        momentum_schedule = self.cosine_scheduler(self.args.schedule, 1, 1, len(test_dataloader))
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        for iiter, (image, label, _) in tqdm(enumerate(test_dataloader), desc=f"TTA for {t}", total=len(test_dataloader)):
            image = image.to(device)
            label = label.to(device)
            ground_truth_batch = torch.arange(len(image), dtype=torch.long, device=self.args.device)
            # ground_truth_txts = torch.arange(self.num_classes, dtype=torch.long, device=self.args.device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if t <= task:  # update average accuracy for current batch
                if self.args.oracle:
                    text_features_predicted = text_features[label]
                else:
                    logits_per_image_predicted_phase_1 = model.logit_scale.exp() * image_features @ text_features.T
                    predicted_class = torch.argmax(logits_per_image_predicted_phase_1, dim=1)
                    text_features_predicted = text_features[predicted_class]

                logits_per_image_phase_2 = torch.mul(model.logit_scale.exp() * image_features @ text_features_predicted.T, 100)
                logits_per_text_phase_2 = logits_per_image_phase_2.T
                total_loss = (loss_img(logits_per_image_phase_2, ground_truth_batch) + loss_txt(logits_per_text_phase_2, ground_truth_batch))/2.0
                total_loss.backward()
                loss.update(total_loss.item() / image.shape[0], n=image.shape[0])
                self.update_model(model, optimizer, count=image.shape[0], task=t)
                optimizer.zero_grad()
                if self.args.ema:
                    with torch.no_grad():
                        m = momentum_schedule[iiter]
                        for param_q, param_k in zip(model.parameters(), teacher_model.parameters()):
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                if iiter % self.args.print_frequency == 0:
                    print(f'TTA Loss {loss.val:.4f} ({loss.avg: .4f}) \t')

class FinetuneFFN(FinetuneCLIP):
    def unfreeze_model(self, model):
        model.train()
        for name, param in model.named_parameters():
            if self.args.finetune_proj:
                trainable_params = (
                    'c_proj' in name and 'visual' in name) or name == 'visual.proj'
            else:
                trainable_params = 'c_proj' in name and 'visual' in name
            if trainable_params:
                param.requires_grad = True
            else:
                param.requires_grad = False


class FinetuenProj(FinetuneCLIP):
    def unfreeze_model(self, model):
        model.train()

        for name, param in model.named_parameters():
            if self.args.finetune_proj:
                trainable_params = (
                    'c_proj' in name and 'visual' in name) or name == 'visual.proj'
            else:
                trainable_params = 'c_proj' in name and 'visual' in name
            if trainable_params:

                param.requires_grad = True
            else:
                param.requires_grad = False


class FinetuenProjTV(FinetuneCLIP):
    def unfreeze_model(self, model):
        model.train()

        for name, param in model.named_parameters():
            if self.args.finetune_proj:
                trainable_params = ('c_proj' in name) or name == 'visual.proj'
            else:
                trainable_params = 'c_proj' in name
            if trainable_params:

                param.requires_grad = True
            else:
                param.requires_grad = False


class FinetuneTextProj(FinetuneCLIP):
    def unfreeze_model(self, model):
        model.train()
        for name, param in model.named_parameters():
            if 'c_proj' in name and 'visual' not in name:
                if self.args.debug:
                    print(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
