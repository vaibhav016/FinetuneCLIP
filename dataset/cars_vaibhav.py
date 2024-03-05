import datasets
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
from dataset.cars_name import classes as car_names
from dataset.cars_name import templates, order
from dataset.cifar100 import SplitCifar100

sys.path.append("..")
from clip.clip import tokenize


def store_paths_with_labels(csv_file, files_list):

    file_dict = {}
    for path in files_list:
        filename = path.split("/")[-1]
        file_dict[str(filename)] = path

    df = pd.read_csv(csv_file)
    paths_with_labels = []
    for i, j in tqdm(df.iterrows()):
        paths_with_labels.append((file_dict[j["filename"]], j["Labels"] - 1))

    return paths_with_labels


def add_header(input_file, output_file):
    # Define the header row as a list
    header = ["filename", "bb1", "bb2", "bb3", "bb4", "Labels"]
    # Replace with your desired column names

    # Specify the input CSV file and output CSV file

    # Read the original CSV file and store its content in a list
    data = []
    with open(input_file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    # Add the header row at the beginning of the data list
    data.insert(0, header)

    # Write the updated data back to the CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def download_transform_data(data_directory):
    if os.path.isdir(os.path.join(data_directory, "cars")):
        print("Cars data downloaded")
    else:
        print("Download data from https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder")  # TODO implement data downloading
    data_directory = os.path.join(data_directory, "cars")
    images_dir = os.path.join(data_directory, "car_data/car_data/")
    train_images = glob.glob(images_dir + "train/*/*.jpg")
    test_images = glob.glob(images_dir + "test/*/*.jpg")

    training_csv_path = os.path.join(data_directory, "anno_train.csv")
    test_csv_path = os.path.join(data_directory, "anno_test.csv")

    training_csv_path_out = os.path.join(data_directory, "anno_train_header.csv")
    test_csv_path_out = os.path.join(data_directory, "anno_test_header.csv")

    add_header(training_csv_path, training_csv_path_out)
    add_header(test_csv_path, test_csv_path_out)

    training_paths = store_paths_with_labels(training_csv_path_out, train_images)
    test_paths = store_paths_with_labels(test_csv_path_out, test_images)

    return training_paths, test_paths


def build_data_loaders(data_directory, config):
    training_paths, test_paths = download_transform_data(data_directory)
    random.shuffle(test_paths)
    test_files_halved = int(len(test_paths) / 2)

    ### Overiding tasks, classes_per_tasks and output_classes to avoid mistakes during runtime in config file
    config.data_config["tasks"] = 14
    config.data_config["classes_per_task"] = 14
    config.train_config["output_classes"] = 196

    # Halve the test data. One half will go for testing, and other half will go for tta
    tta_paths = test_paths[:test_files_halved]
    test_eval_paths = test_paths[test_files_halved:]

    if config.train_config["training_mode"] == "cl":
        return cl_data(training_paths, test_eval_paths, Cars, config)
    elif config.train_config["training_mode"] == "joint":
        return joint_data(training_paths, test_eval_paths, Cars, config)
    elif config.train_config["training_mode"] in ["tta_cl", "tta_cl_rmt", "lwf", "ematsp", "tta_cl_shot", "tta_cl_shot_dino_supervised",
                                                  "tta_cl_shot_dino"]:
        return tta_cl_data(training_paths, test_eval_paths, tta_paths, Cars, config)
    else:
        raise Exception("Incorrect training mode")

class Cars(Dataset):
    def __init__(self, path_dict_with_labels, transforms=None):
        # transforms: none  # Can be 1) autoaugment, 2) augmix, 3) randaugment, 4) trivialaugment, 5) manual, 6) none
        self.image_paths_dict_with_labels = path_dict_with_labels
        self.transform = transforms

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
        name = car_names[label]
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
        name = car_names[label]
        text = tokenize(f'a photo of a {name}')[0]
        return image, label, text


class SplitCarsV(SplitCifar100):
    def __init__(self, args, root='./', transform=None):
        # root = '/Users/vaibhavsingh/Desktop/concordia_phd/TTA_datasets'
        training_paths, test_paths = download_transform_data(args.data)
        random.shuffle(test_paths)
        test_files_halved = int(len(test_paths) / 2)

        # Halve the test data. One half will go for testing, and other half will go for tta
        tta_paths = test_paths[:test_files_halved]
        test_eval_paths = test_paths[test_files_halved:]

        # context_feat = Features({'text': Value(dtype='string', id=None)})
        self.trainset = Cars(training_paths)
        self.testset = Cars(test_eval_paths)
        self.ttaset = Cars(tta_paths)

        self.trainset.targets = [i[1] for i in self.trainset]
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]

        self.transform = transform
        self.root = root

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = 196
        self.num_tasks = 10
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(240 * args.buffer_size)
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}

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

        self.class_name_full = car_names

        self.classifier = zeroshot_classifier


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
