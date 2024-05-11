import copy
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, FGVCAircraft

from dataset.aircraft_name import classes as class_names_aircraft


from tqdm import tqdm

from clip.clip import tokenize
from dataset.cifar100_name import classes as class_names_cifar100
from dataset.cifar100_name import templates, order
from dataset.aircraft_name import templates2
from torch.utils.data import random_split


def zeroshot_classifier(classnames, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zeroshot_weights = []
    with torch.no_grad():
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates2]  # format with class
            texts = tokenize(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(
                texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.T

class CLIPDataset(Dataset):
    _repr_indent = 4

    def __init__(self, set, text, idx, **kwargs):
        self.data = set
        self.text = text
        self.idx = idx
        self.classes = np.array([self.data.targets[i] for i in idx])

    def __len__(self):
        return len(self.idx)

    def __repr__(self) -> str:

        head = "Dataset "
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += [f"Current seen classes {len(np.unique(self.classes))}"]
        body.append("Image root location: {}".format(self.data.root))
        if hasattr(self.data, "transform") and self.data.transform is not None:
            body += [repr(self.data.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __getitem__(self, idx):
        index = self.idx[idx]
        image, label = self.data[index]
        name = self.text[label]
        name = name.replace('_', ' ')
        text = tokenize(f'a photo of {name}')[0]

        return image, label, text

class LongSequence():
    def __init__(self, args, transform=None):
        self.training_tasks = []
        self.tta_tasks = []
        self.testset_tasks = []
        self.class_names_list = []
        self.buffer = {}
        self.paths = {}
        self.args = args
        self.scenario = "dataset_incremental"
        self.buffer_size = int(args.buffer_size)
    
        self.add_cifar_data(transform)    ########## CIFAR100##############
        self.add_aircraft_data(transform) ########## Aircraft #############
        
        self.dataset_collect_fcn = CLIPDataset
        self.classifier = zeroshot_classifier
        self.num_tasks = len(self.training_tasks)
        self.num_classes = 10
        self.task = 0
        
        
    def add_aircraft_data(self,transform):
        root = self.args.data + "/aircrafts_data/fgvc-aircraft-2013b"
        self.trainset_aircraft = FGVCAircraft(
            root, split='train', transform=transform,download=True)
        self.testset_aircraft = FGVCAircraft(root, split='test', transform=transform,download=True)
        self.ttaset_aircraft =  FGVCAircraft(root, split='val', transform=transform,download=True)
        
        self.trainset_aircraft.targets = self.trainset_aircraft._labels
        self.testset_aircraft.targets = self.testset_aircraft._labels
        self.ttaset_aircraft.targets = self.ttaset_aircraft._labels

        classnames_aircraft = class_names_aircraft

        self.training_tasks.append(self.trainset_aircraft)
        self.tta_tasks.append(self.ttaset_aircraft)
        self.testset_tasks.append(self.testset_aircraft)
        
        self.class_names_list.append(classnames_aircraft)

    def add_cifar_data(self, transform):
        self.trainset_cifar100 = CIFAR100(
            self.args.data, train=True, transform=transform, download=True)
        self.testset_cifar100 = CIFAR100(
            self.args.data, train=False, transform=transform, download=True)
        self.transform = transform
        
        test_data_len = len(self.testset_cifar100)
        self.class_names_list.append(class_names_cifar100)
        
        self.ttaset, self.testset = random_split(self.testset_cifar100, [test_data_len // 2, test_data_len - (test_data_len // 2)])
        
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]

        self.training_tasks.append(self.trainset_cifar100)
        self.tta_tasks.append(self.ttaset)
        self.testset_tasks.append(self.testset)
    
    # def add_cars_data(self, transform):
    #     #################### CARS #########################
    #     training_paths, test_paths = download_transform_data(args.data)
    #     random.shuffle(test_paths)
    #     test_files_halved = int(len(test_paths) / 2)

    #     # Halve the test data. One half will go for testing, and other half will go for tta
    #     tta_paths = test_paths[:test_files_halved]
    #     test_eval_paths = test_paths[test_files_halved:]

    #     # context_feat = Features({'text': Value(dtype='string', id=None)})
    #     self.trainset = Cars(training_paths)
    #     self.testset = Cars(test_eval_paths)
    #     self.ttaset = Cars(tta_paths)

    #     self.trainset.targets = [i[1] for i in self.trainset]
    #     self.ttaset.targets = [i[1] for i in self.ttaset]
    #     self.testset.targets = [i[1] for i in self.testset]
        

    def get_dataset(self, task_id, is_train=True, with_buffer=True, balanced=False, is_tta=False):
        if is_train:
            self.mode = 'train'
            self.set = self.training_tasks[task_id]
        else:
            if is_tta:
                self.mode = "tta"
                self.set = self.tta_tasks[task_id]
            else:
                self.mode = 'test'
                self.set = self.testset_tasks[task_id]

        self._get_image_list_for_cur_set(with_buffer=with_buffer)
        idx = copy.deepcopy(self.data_idx)
        curset = self.dataset_collect_fcn(
            self.set, self.class_names_list[task_id], idx, transform=self.transform)

        return curset
    
    def _get_image_list_for_cur_set(self, with_buffer=True):
        if f'{self.task}_{self.mode}' in self.paths.keys():
            # if we have already read paths and labels from fils
            self.data_idx = self.paths[f'{self.task}_{self.mode}']['data_idx']
        else:
            targets = self.set.targets
            # prepare data idx for current set
            self.data_idx = []
            for idx in range(len(targets)):
                    self.data_idx.append(idx)

            # and save to self.path
            self.paths[f'{self.task}_{self.mode}'] = {}
            self.paths[f'{self.task}_{self.mode}']['data_idx'] = self.data_idx

        if (self.buffer is not {}) and (self.mode == 'train') and (with_buffer):
            for key in self.buffer.keys():
                self.data_idx.extend(self.buffer[key]['data_idx'])
 
    def get_buffer(self, task):
        self.set = self.training_tasks[task]
        assert (task-1 in self.buffer.keys())
        self.data_idx = []
        for key in range(task):
            self.data_idx.extend(self.buffer[key]['data_idx'])
        idx = copy.deepcopy(self.data_idx)
        return self.dataset_collect_fcn(self.set, self.class_names_list[task], idx, transform=self.transform, n=self.shot)
    
    def update_buffer(self, task):
        '''
        update the buffer with data from task,
        the buffer is task balanced such that
        it always contains the same number of
        samples from different tasks.
        :param task: update buffer with samples from which task,

        '''
        # make sure we have already read path file for the task and did not update buffer with this task
        assert (f'{task}_train' in self.paths.keys()) and (
            task not in self.buffer.keys())
        cur_buffer_task_length = self.buffer_size // (
            len(self.buffer.keys()) + 1)

        # cut buffer sample from previous task
        for key in self.buffer.keys():
            pre_task_length = len(self.buffer[key]['data_idx'])
            cur_task_length = min(cur_buffer_task_length, pre_task_length)
            indices = random.sample(range(pre_task_length), cur_task_length)
            self.buffer[key]['data_idx'] = [
                self.buffer[key]['data_idx'][i] for i in indices]
        # update buffer with current task
        task_length = len(self.paths[f'{task}_train']['data_idx'])
        cur_task_length = min(cur_buffer_task_length, task_length)
        indices = random.sample(range(task_length), cur_task_length)

        key = task
        self.buffer[key] = {}
        self.buffer[key]['data_idx'] = [
            self.paths[f'{task}_train']['data_idx'][i] for i in indices]
        self.current_buffer_length = sum(
            [len(self.buffer[key]['data_idx']) for key in self.buffer.keys()])

