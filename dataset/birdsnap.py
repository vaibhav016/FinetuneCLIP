import sys
import os
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset.birdsnap_name import classes as class_names
from dataset.birdsnap_name import templates
from dataset.birdsnap_name import order
from dataset.cifar100 import CLIPDataset, SplitCifar100, FewShotCLIPDataset


sys.path.append("..")
from clip.clip import tokenize

class SplitBirdsnap(SplitCifar100):
    def __init__(self, args, root='./', transform=None, valid=False, num_tasks=10):
        self.root = root
        self.set = ImageFolder(self.root, transform=transform)

        self.idx_dir = './'
        with open(f'{self.idx_dir}/test_images.txt', 'r') as f:
            lines = f.readlines()
            test_paths = [os.path.join(self.root, line.strip())
                          for line in lines if 'jpg' in line]

        # Create a list of indices corresponding to test samples
        test_indices = [idx for idx, (path, _) in enumerate(
            self.set.samples) if path in test_paths]


        # Create a list of indices corresponding to training samples
        train_indices = [idx for idx in range(
            len(self.set)) if idx not in test_indices]
        self.trainset = Subset(self.set, train_indices)
        self.testset = Subset(self.set, test_indices)
        print(len(self.trainset), len(self.testset))
        self.testset.targets = [self.set.targets[i]
                                for i in self.testset.indices]
        self.trainset.targets = [self.set.targets[i]
                                 for i in self.trainset.indices]
        self.trainset.root = './'
        self.testset.root = './'
        self.transform = transform

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = len(class_names)

        self.num_tasks = num_tasks if not args.joint else 1

        self.buffer_size = int(args.buffer_size)
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}

        self.class_name_full = class_names
        self.classifier = zeroshot_classifier

        classes = order

        self.task_classes = np.array_split(classes, self.num_tasks)
        print('task split', self.task_classes)

        if args.few_shot > 0:
            self.dataset_collect_fcn = FewShotCLIPDataset
            self.shot = args.few_shot
        else:
            self.dataset_collect_fcn = CLIPDataset
            self.shot = 0

        self._get_image_list_for_cur_set()


def zeroshot_classifier(classnames, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            texts = tokenize(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(
                texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.T
