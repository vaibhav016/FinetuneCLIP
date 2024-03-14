import glob
import os
import random
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataset.pets_name import order
from dataset.cifar100 import SplitCifar100
from dataset.pets_name import classes as pet_names

sys.path.append("..")
from clip.clip import tokenize

class Pets(Dataset):
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


def store_paths_with_labels(all_images, train_annotation_file, test_annotation_file):

    images_2_label_train = {}
    with open(train_annotation_file) as f:
        lines = f.readlines()
    for m in lines:
        k = m.split(" ")
        images_2_label_train[k[0]] = int(k[1])

    images_2_label_test = dict()
    with open(test_annotation_file) as f:
        lines = f.readlines()
    for m in lines:
        k = m.split(" ")
        images_2_label_test[k[0]] = int(k[1])

    train_paths_with_labels = []
    test_paths_with_labels = []

    for file in all_images:
        breed = file.split("/")[-1].split(".")[0]
        if breed in images_2_label_train:
            train_paths_with_labels.append((file, images_2_label_train[breed] - 1))
        elif breed in images_2_label_test:
            test_paths_with_labels.append((file, images_2_label_test[breed] - 1))
        else:
            continue

    return train_paths_with_labels, test_paths_with_labels


def download_transform_data(data_directory):
    if os.path.isdir(os.path.join(data_directory, "images")) and os.path.isdir(os.path.join(data_directory, "annotations")):
        print("Pets data downloaded")
    else:
        print("https://www.robots.ox.ac.uk/~vgg/data/pets/")  # TODO implement data downloading
    # data_directory = os.path.join(data_directory, "images")
    images_dir = os.path.join(data_directory, "images")
    all_images = glob.glob(images_dir + "/*.jpg")
    train_annotation_file = os.path.join(data_directory, "annotations/trainval.txt")
    test_annotation_file = os.path.join(data_directory, "annotations/test.txt")

    training_paths, test_paths = store_paths_with_labels(all_images, train_annotation_file, test_annotation_file)
    print(len(training_paths), len(test_paths))

    return training_paths, test_paths


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
        # label = int(self.data[index]['label'])
        image, label = self.data[index]
        # image = self.transform(image)
        name = pet_names[label]
        text = tokenize(f'a photo of a {name}')[0]
        return image, label, text

class FewShotCLIPDataset(Dataset):
    _repr_indent = 4

    def __init__(self, set, text, idx, n, transform, **kwargs):
        self.data = set
        self.text = text
        self.n = n
        self.transform = transform
        self.classes = np.array([self.data.targets[i] for i in idx])
        self.data.transform = transform
        self.idx = self. _sample_n_samples_per_class()

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

    def __getitem__(self, idx):
        index = int(self.idx[idx])
        image = self.transform(self.data[index]['image'])
        label = int(self.data[index]['label'])
        name = pet_names[label]
        text = tokenize(f'a photo of a {name}')[0]
        return image, label, text


class SplitPetsV(SplitCifar100):
    def __init__(self, args, root='./', transform=None):
        root = args.data
        training_paths, test_paths = download_transform_data(args.data)
        random.shuffle(test_paths)
        test_files_halved = int(len(test_paths) / 2)

        # Halve the test data. One half will go for testing, and other half will go for tta
        tta_paths = test_paths[:test_files_halved]
        test_eval_paths = test_paths[test_files_halved:]

        self.trainset = Pets(training_paths)
        self.testset = Pets(test_eval_paths)
        self.ttaset = Pets(tta_paths)

        self.trainset.targets = [i[1] for i in self.trainset]
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]

        self.transform = transform
        self.root = root

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = 37
        self.num_tasks = 10
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(args.buffer_size)
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}

        self.class_name_full = pet_names
        print(self.class_name_full)
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
