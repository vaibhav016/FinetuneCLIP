from dataset.cifar100 import SplitCifar100
import datasets
from clip.clip import tokenize
from torch.utils.data import Dataset
import numpy as np
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from PIL import Image
from datasets import load_dataset,Features,Value
import csv
import os
import glob
import random
from dataset.cars_name import templates, order
from dataset.cifar100 import SplitCifar100

order = [51,  94,  55,  35, 170, 177,  20,  85,  50,  36,  30,  76,   5, 136, 182,  82,  25, 169, 166, 178,
         74,  53,  32, 184, 160, 179, 138, 140,  27,  12,  48,  57, 145,  28,  19, 162, 175, 121,  18,  72,
         101,  69,  49, 115, 181,  15, 193,  37, 111,   0, 158,  33,  11,  47,  80, 126, 183,  16, 198,  91,
         58,  70,   2,  67,   8, 199,  10,   3,  77,  22, 168,  96,  86,   4, 189,  88,  99,  31,  84,  17,
         107, 123,  29, 103, 117, 161, 105,  73, 173,  13,  24,   1, 195, 185,  79,  87, 151,  65,  62,  26,
         147, 144,  52,  75, 186, 159, 109,  66, 137, 191, 122, 133, 142,  38,  39,  61,  98, 157, 192, 129,
         112, 197, 149, 194, 104, 152, 120,  56, 124, 132,  89, 141, 116, 146, 153, 176, 127,  71, 125,  63,
         135, 118, 102,  41, 150, 154,  90, 172, 167, 106, 114,  46, 165, 131, 196, 156, 180,  34,  44,  83,
         164,   6,  59,  60,  45, 143,  42, 134, 108,  97,  81, 119,  93,   7, 187,  68, 128, 113, 139,  95,
         130, 100, 163, 110,  40, 174, 148,   9, 190,  54, 155,  64,  78, 171, 188,  43,  92,  21,  23,  14]

cub_name = []
with open('./dataset/cub_name.txt', 'r') as f:
    for line in f.readlines():
        name = line.strip().split('.')[-1].replace("_", ' ')
        cub_name.append(name)


class CLIPDataset(Dataset):
    _repr_indent = 4

    def __init__(self, set, text, idx, transform, **kwargs):
        self.data = set
        self.text = text
        self.idx = idx
        self.transform = transform
        self.classes = np.array([self.data.targets[i] for i in idx])
        self.data.transform = transform

    def __len__(self):
        return len(self.idx)

    def __repr__(self) -> str:
        head = "Dataset "
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += [f"Current seen classes {len(np.unique(self.classes))}"]
        if hasattr(self.data, "transform") and self.data.transform is not None:
            body += [repr(self.data.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __getitem__(self, idx):
        index = int(self.idx[idx])
        # image = self.transform(self.data[index]['image'])
        # # name = self.data[index]['clip_tags_ViT_B_16_simple_specific']
        # label = int(self.data[index]['label'])
        image, label = self.data[index]
        name = cub_name[label]
        text = tokenize(f'a photo of a {name}')[0]
        return image, label, text


class FewShotCLIPDataset(Dataset):
    _repr_indent = 4

    def __init__(self, set, text, idx, n, transform, **kwargs):
        self.data = set
        self.text = text
        self.n = n
        # self.idx = idx
        self.transform = transform
        self.classes = np.array([self.data.targets[i] for i in idx])
        self.data.transform = transform
        self.idx = self._sample_n_samples_per_class()

    def _sample_n_samples_per_class(self):
        sampled_indices = []
        for class_label in np.unique(self.classes):
            class_indices = np.where(self.classes == class_label)[0]
            if len(class_indices) <= self.n:
                sampled_indices.extend(class_indices)
            else:
                sampled_indices.extend(np.random.choice(
                    class_indices, self.n, replace=False))
        return sampled_indices

    def __len__(self):
        return len(self.idx)

    def __repr__(self) -> str:
        head = "Dataset "
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += [f"Current seen classes {len(np.unique(self.classes))}"]
        if hasattr(self.data, "transform") and self.data.transform is not None:
            body += [repr(self.data.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __getitem__(self, idx):
        index = int(self.idx[idx])
        image = self.transform(self.data[index]['image'])
        label = int(self.data[index]['label'])
        name = cub_name[label]
        text = tokenize(f'a photo of a {name}')[0]
        return image, label, text


class Cub(Dataset):
    def __init__(self, path_dict_with_labels, transform=None, config=None, resize_image=224):
        # transforms: none  # Can be 1) autoaugment, 2) augmix, 3) randaugment, 4) trivialaugment, 5) manual, 6) none
        self.image_paths_dict_with_labels = path_dict_with_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths_dict_with_labels)

    def __getitem__(self, idx):
        data = self.image_paths_dict_with_labels[idx]
        # print(data)
        image = Image.open(data[0])
        # if image.mode != "RGB":
        #     # Convert grayscale to RGB by duplicating the single channel
        #     image = image.convert("RGB")
        image = self.transform(image) if self.transform is not None else image
        labels = data[1]

        return image, labels


def download_transform_data(data_directory):
    if os.path.isdir(os.path.join(data_directory, "CUB_200_2011")):
        print("CUB data downloaded")
    else:
        print("Download data from https://data.caltech.edu/records/65de6-vp158")  # TODO implement data downloading
    data_directory = os.path.join(data_directory, "CUB_200_2011")
    images_dir = os.path.join(data_directory, "images/")

    df = pd.read_csv(os.path.join(data_directory, "images.txt"), sep=" ", header=None, names=["image_id", "image_name"], )
    df["class_name"] = df["image_name"].apply(lambda x: x.split("/")[0])
    df["image_name"] = df["image_name"].apply(lambda x: os.path.join(images_dir, x))

    df2 = pd.read_csv(os.path.join(data_directory, "classes.txt"), sep=" ", header=None, names=["label", "class_name"], )

    df3 = pd.read_csv(os.path.join(data_directory, "image_class_labels.txt"), sep=" ", header=None, names=["image_id", "class_id"], )

    df4 = pd.read_csv(os.path.join(data_directory, "train_test_split.txt"), sep=" ", header=None, names=["image_id", "is_train"], )

    df = df.merge(df2, on="class_name", how="inner")
    df = df.merge(df3, on="image_id", how="inner")
    df = df.merge(df4, on="image_id", how="inner")

    training_paths = []
    test_paths = []
    for i, j in tqdm(df.iterrows()):
        if j["is_train"] == 1:
            training_paths.append((j["image_name"], j["class_id"] - 1))
        else:
            test_paths.append((j["image_name"], j["class_id"] - 1))

    print(len(training_paths), len(test_paths))
    return training_paths, test_paths


def build_data_loaders(data_directory):
    training_paths, test_paths = download_transform_data(data_directory)
    random.shuffle(test_paths)
    test_files_halved = int(len(test_paths) / 2)

    ### Overiding tasks, classes_per_tasks and output_classes to avoid mistakes during runtime in config file
    config.data_config["tasks"] = 20
    config.data_config["classes_per_task"] = 10
    config.train_config["output_classes"] = 200

    # Halve the test data. One half will go for testing, and other half will go for tta
    tta_paths = test_paths[:test_files_halved]
    test_eval_paths = test_paths[test_files_halved:]

    if config.train_config["training_mode"] == "cl":
        return cl_data(training_paths, test_eval_paths, Cub, config)
    elif config.train_config["training_mode"] == "joint":
        return joint_data(training_paths, test_eval_paths, Cub, config)
    elif config.train_config["training_mode"] in ["tta_cl", "tta_cl_rmt", "lwf", "ematsp", "tta_cl_shot", "tta_cl_shot_dino_supervised",
                                                  "tta_cl_shot_dino"]:
        return tta_cl_data(training_paths, test_eval_paths, tta_paths, Cub, config)
    else:

        raise Exception("Incorrect training mode")


class CUBV(SplitCifar100):
    def __init__(self, args, root='./', transform=None):
        root = args.data
        training_paths, test_paths = download_transform_data(args.data)
        random.shuffle(test_paths)
        test_files_halved = int(len(test_paths) / 2)

        # Halve the test data. One half will go for testing, and other half will go for tta
        tta_paths = test_paths[:test_files_halved]
        test_eval_paths = test_paths[test_files_halved:]

        self.trainset = Cub(training_paths)
        self.testset = Cub(test_eval_paths)
        self.ttaset = Cub(tta_paths)

        self.trainset.targets = [i[1] for i in self.trainset]
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]

        self.transform = transform
        self.root = root

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = 200
        self.num_tasks = 10
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(240 * args.buffer_size)
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}

        self.get_full_class_name(self.trainset)
        print(self.class_name_full)
        # self.trainset.targets = self.trainset['label']
        # self.testset.targets = self.testset['label']

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

    def get_full_class_name(self, set):

        self.class_name_full = cub_name
