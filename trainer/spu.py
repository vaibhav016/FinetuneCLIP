import copy
import re
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from trainer.finetune import FinetuneCLIP
import os


def zerolike_params_dict(model, device=None):
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    """

    return [
        (k, torch.zeros_like(p).to(p.device if (device == None) else device))
        for k, p in model.named_parameters()
    ]


def copy_params_dict(model, copy_grad=False, device=None):
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [(k, p.grad.data.detach().clone()) for k, p in model.named_parameters()]
    else:
        return [(k, p.data.detach().clone().to(p.device if (device == None) else device)) for k, p in
                model.named_parameters()]


class MASEDIT(FinetuneCLIP):
    def __init__(self, args):
        super().__init__(args)
        self.magnitudes = {}
        self.mask = {}
        self.ttl_mask = {}
        self.alpha = 0.5
        self._lambda = self.args.scale
        self.importance_computed = False
        self.trainable_params = []
        self.mask_per_task = {i:{} for i in range(self.args.num_tasks)}
        self.ttl_mask_per_task = {i:{} for i in range(self.args.num_tasks)}
        self.mask_per_task_union = {i:{} for i in range(self.args.num_tasks)}


    def setup_importance(self, model):
        # Parameters before the first task starts
        self.params = dict(copy_params_dict(model))
        # Initialize Fisher information weight importance
        self.importance = dict(zerolike_params_dict(model))

    def unfreeze_model(self, model):
        model.train()
        for name, param in model.named_parameters():
            if self.args.update_all:
                trainable_params = True
            elif any(edit_layer in name for edit_layer in ['c_fc', 'visual.proj']):
                trainable_params = True
            else:
                trainable_params = False

            if trainable_params:
                param.requires_grad = True
                if name not in self.trainable_params:
                    self.trainable_params.append(name)
            else:
                param.requires_grad = False
        # print('Trainable parameters: ', self.trainable_params)

    def compute_importance(self, dataset, model, task):
        if task == 0:
            self.setup_importance(model)

        cur_set = dataset.get_dataset(task, is_train=True, with_buffer=False)
        loader = self.get_loader(cur_set, is_train=True)
        print('Compute importance for the current task...')
        cur_importance = self.compute_importance_score(model, loader, loss_type=self.args.select_loss_type, task=task,
                                                       dataset=dataset)

        if self.args.save_ckpt:
            with open(os.path.join(self.args.log_path, f'task{task}_importance.torchSave'), 'wb') as file:
                torch.save(cur_importance, file)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.trainable_params:
                    if any(param in name for param in ['bias']) or self.args.sparsity == 1.0:
                        # self.mask[name] = torch.ones(param.shape, dtype=param.dtype).to(self.args.device)
                        self.mask_per_task[task][name] = torch.ones(param.shape, dtype=param.dtype).to(self.args.device)

                        continue
                    if name not in cur_importance.keys():
                        print(f' importance of `{name} is none')
                        continue
                    importance = cur_importance[name]

                    # sparse update of weight and  bias
                    if self.args.score == 'norm':
                        magnitudes = importance.abs()
                    elif self.args.score == 'random':
                        magnitudes = torch.randn(param.grad.shape).to(self.args.device)
                    else:
                        raise ValueError

                    k = int(magnitudes.numel() * self.args.sparsity)
                    # print("magnitudes-----------",magnitudes.shape)
                    # print("k-----", k)

                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    # print("topkvaluse-------", topk_values.shape, topk_indices)
                    
                    # self.mask[name] = torch.zeros_like(magnitudes).to(self.args.device)
                    self.mask_per_task[task][name] = torch.zeros_like(magnitudes).to(self.args.device)
                    # print("mask--------------", self.mask[name].shape)
                    # self.mask[name].view(-1)[topk_indices] = 1
                    self.mask_per_task[task][name].view(-1)[topk_indices] = 1
                    
    def update_model(self, model, optimizer, task, **kwargs):
        # print("----------- spu update --------------")
        count = kwargs.get('count', 0)
        epoch = kwargs.get('epoch', 0)
        with torch.no_grad():
            for name, param in model.named_parameters():
                gradients = param.grad
                if gradients is not None:
                    if self.args.supervised_union_masks_per_task:
                        param.grad =  self.mask_per_task_union[task][name] * param.grad
                    else:
                        param.grad =  self.mask_per_task[task][name] * param.grad
                    # Update only the 1% most activated entries
                    # param.data -= optimizer.param_groups[0]['lr'] * param.grad
        optimizer.step()

    def compute_loss(self, batch, teacher_model, model, task, **kwargs):
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
        ground_truth = torch.arange(len(images), dtype=torch.long, device=self.args.device)

        logits_per_image, logits_per_text = model(images, texts)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        if self.args.rd_loss and task>0: #it does not make sense to put rd loss for 1st task
            with torch.no_grad():
                logits_per_image_teacher, logits_per_text_teacher = teacher_model(images, texts)
        
            rd_loss = self.kl_div_loss(logits_per_text, logits_per_text_teacher)
            total_loss+=rd_loss
        
        return total_loss

    def compute_importance_score(self, model, dataloader, loss_type='l2', **kwargs):
        # Initialize importance matrix
        importance = dict(zerolike_params_dict(model, device=self.args.device))

        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        model.zero_grad()
        total_batch = len(dataloader)
        num_batch_for_importance = total_batch * self.args.cur_importance_batch_percentage
        print(f'Total batch for importance {total_batch}, use {num_batch_for_importance} batches')
        stop_flag = 1

        for num_batch, batch in enumerate(tqdm(dataloader)):
            stop_flag = 1
            # Get batch
            images, _, texts = batch
            images = images.to(self.args.device)
            texts = texts.to(self.args.device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Average L2-Norm of the output
            if loss_type == 'l2':
                loss = torch.norm(logits_per_image, p="fro", dim=1).pow(2).mean()
            elif loss_type == 'cn':
                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.args.device)
                loss_img = nn.CrossEntropyLoss()
                loss_txt = nn.CrossEntropyLoss()
                loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            else:
                raise ValueError
            loss.backward()

            # Accumulate importance
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        importance[name].data += param.grad.clone()
                        if importance[name].data.abs().min() < 1e-12:
                            stop_flag = 0
            if num_batch > num_batch_for_importance and stop_flag:
                break

        return importance