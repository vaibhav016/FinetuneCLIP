
from clip.clip import tokenize

import sys

import torch
from tqdm import tqdm
import numpy as np

from torchvision.datasets import FGVCAircraft
from dataset.aircraft_name import classes as class_names
from dataset.aircraft_name import templates, order, templates2
from dataset.cifar100 import SplitCifar100, CLIPDataset, FewShotCLIPDataset

sys.path.append("..")


class SplitAircraft(SplitCifar100):
    def __init__(self, args, root='./', transform=None):
        root = './'
        self.trainset = FGVCAircraft(
            root, split='train', transform=transform,download=True)
        self.testset = FGVCAircraft(root, split='test', transform=transform,download=True)
        self.ttaset =  FGVCAircraft(root, split='val', transform=transform,download=True)
        self.trainset.targets = self.trainset._labels
        self.testset.targets = self.testset._labels
        self.ttaset.targets = self.ttaset._labels

        self.transform = transform
        self.root = root

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = 100
        self.num_tasks = 10
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(args.aircraft_buffer_size * args.buffer_size) # default buffer size=250
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}
        self.class_to_idx = self.trainset.class_to_idx
        self.class_name_full = class_names
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

        self.classifier = zeroshot_classifier


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
