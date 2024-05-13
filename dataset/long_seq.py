import copy
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, FGVCAircraft, GTSRB
from PIL import Image
from dataset.aircraft_name import classes as class_names_aircraft
from dataset.cars_name import classes as car_names
import pandas as pd
import csv
import os
import glob

from tqdm import tqdm

from clip.clip import tokenize
from dataset.cifar100_name import classes as class_names_cifar100
from dataset.cifar100_name import templates, order
from dataset.aircraft_name import templates2
from torch.utils.data import random_split
from dataset.pets_name import classes as pet_names
from dataset.gtsrb_name import classes as gtsrb_names



order_cub = [51,  94,  55,  35, 170, 177,  20,  85,  50,  36,  30,  76,   5, 136, 182,  82,  25, 169, 166, 178,
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

def download_transform_data_cars(data_directory):
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

def download_transform_data_cub(data_directory):
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

def store_paths_with_labels_pets(all_images, train_annotation_file, test_annotation_file):

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

def download_transform_data_pets(data_directory):
    if os.path.isdir(os.path.join(data_directory, "images")) and os.path.isdir(os.path.join(data_directory, "annotations")):
        print("Pets data downloaded")
    else:
        print("https://www.robots.ox.ac.uk/~vgg/data/pets/")  # TODO implement data downloading
    # data_directory = os.path.join(data_directory, "images")
    images_dir = os.path.join(data_directory, "images")
    all_images = glob.glob(images_dir + "/*.jpg")
    train_annotation_file = os.path.join(data_directory, "annotations/trainval.txt")
    test_annotation_file = os.path.join(data_directory, "annotations/test.txt")

    training_paths, test_paths = store_paths_with_labels_pets(all_images, train_annotation_file, test_annotation_file)
    print(len(training_paths), len(test_paths))

    return training_paths, test_paths


class Pets(Dataset):
    def __init__(self, path_dict_with_labels, transform=None, root = None, config=None, resize_image=224):
        # transforms: none  # Can be 1) autoaugment, 2) augmix, 3) randaugment, 4) trivialaugment, 5) manual, 6) none
        self.image_paths_dict_with_labels = path_dict_with_labels
        self.transform = transform
        self.root = root

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

class Cub(Dataset):
    def __init__(self, path_dict_with_labels, transform=None,  root = None, config=None, resize_image=224):
        # transforms: none  # Can be 1) autoaugment, 2) augmix, 3) randaugment, 4) trivialaugment, 5) manual, 6) none
        self.image_paths_dict_with_labels = path_dict_with_labels
        self.transform = transform
        self.root = root

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

    def __init__(self, set, text, idx, **kwargs):
        self.data = set
        self.text = text
        self.idx = idx
        # print(len(self.data.targets))
        # print(len(idx))
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

class Cars(Dataset):
    def __init__(self, path_dict_with_labels, transforms=None, root=None):
        # transforms: none  # Can be 1) autoaugment, 2) augmix, 3) randaugment, 4) trivialaugment, 5) manual, 6) none
        self.image_paths_dict_with_labels = path_dict_with_labels
        self.transform = transforms
        self.root = root

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

class LongSequence():
    def __init__(self, args, transform=None):
        self.training_tasks = []
        self.tta_tasks = []
        self.testset_tasks = []
        self.class_names_list = []
        self.class_names_extended = []
        self.num_classes = []
        self.buffer = {}
        self.paths = {}
        self.args = args
        self.scenario = "dataset_incremental"
        self.buffer_size = int(args.buffer_size)
        self.transform = transform
    
        self.add_cars_data(transform)    ########## cars ##############
        self.add_aircraft_data(transform) ########## Aircraft #############
        self.add_cub_data(transform)
        self.add_pets_data(transform)
        self.add_gtsrb_data(transform)
        
        # self.add_cifar_data(transform) 
        # Not adding Cifar because it contains classes. 

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
        self.class_names_extended.extend(classnames_aircraft)
        print("---------------- added aircraft data --------------")

    def add_cifar_data(self, transform):
        self.trainset_cifar100 = CIFAR100(
            self.args.data, train=True, transform=transform, download=True)
        self.testset_cifar100 = CIFAR100(
            self.args.data, train=False, transform=transform, download=True)
        self.transform = transform
        
        test_data_len = len(self.testset_cifar100)
        self.class_names_list.append(class_names_cifar100)
        self.class_names_extended.extend(class_names_cifar100)
        
        self.ttaset, self.testset = random_split(self.testset_cifar100, [test_data_len // 2, test_data_len - (test_data_len // 2)])
        
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]

        self.training_tasks.append(self.trainset_cifar100)
        self.tta_tasks.append(self.ttaset)
        self.testset_tasks.append(self.testset)

        print("---------------- added cifar100 data --------------")

    def add_cars_data(self, transform):
        #################### CARS #########################
        training_paths, test_paths = download_transform_data_cars(self.args.data)
        random.shuffle(test_paths)
        test_files_halved = int(len(test_paths) / 2)

        # Halve the test data. One half will go for testing, and other half will go for tta
        tta_paths = test_paths[:test_files_halved]
        test_eval_paths = test_paths[test_files_halved:]

        # context_feat = Features({'text': Value(dtype='string', id=None)})
        self.trainset = Cars(training_paths, transform, root = self.args.data)
        self.testset = Cars(test_eval_paths, transform, root = self.args.data)
        self.ttaset = Cars(tta_paths, transform, root = self.args.data)

        self.trainset.targets = [i[1] for i in self.trainset]
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]

        self.class_names_list.append(car_names)
        self.class_names_extended.extend(car_names)

        self.training_tasks.append(self.trainset)
        self.tta_tasks.append(self.ttaset)
        self.testset_tasks.append(self.testset)
        print("---------------- added cars data --------------")
    
    def add_cub_data(self, transform):
        training_paths, test_paths = download_transform_data_cub(self.args.data)
        random.shuffle(test_paths)
        test_files_halved = int(len(test_paths) / 2)

        # Halve the test data. One half will go for testing, and other half will go for tta
        tta_paths = test_paths[:test_files_halved]
        test_eval_paths = test_paths[test_files_halved:]

        self.trainset = Cub(training_paths, transform, root = self.args.data)
        self.testset = Cub(test_eval_paths, transform, root = self.args.data)
        self.ttaset = Cub(tta_paths, transform, root = self.args.data)

        self.trainset.targets = [i[1] for i in self.trainset]
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]

        self.class_names_list.append(cub_name)
        self.class_names_extended.extend(cub_name)

        self.training_tasks.append(self.trainset)
        self.tta_tasks.append(self.ttaset)
        self.testset_tasks.append(self.testset)
        print("---------------- added cub data --------------")
    
    def add_pets_data(self, transform):
        training_paths, test_paths = download_transform_data_pets(self.args.data)
        random.shuffle(test_paths)
        test_files_halved = int(len(test_paths) / 2)

        # Halve the test data. One half will go for testing, and other half will go for tta
        tta_paths = test_paths[:test_files_halved]
        test_eval_paths = test_paths[test_files_halved:]

        self.trainset = Pets(training_paths, transform, root = self.args.data)
        self.testset = Pets(test_eval_paths, transform, root = self.args.data)
        self.ttaset = Pets(tta_paths, transform, root = self.args.data)

        self.trainset.targets = [i[1] for i in self.trainset]
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]
        
        self.class_names_list.append(pet_names)
        self.class_names_extended.extend(pet_names)

        self.training_tasks.append(self.trainset)
        self.tta_tasks.append(self.ttaset)
        self.testset_tasks.append(self.testset)
        print("---------------- added Pets data --------------")
    
    def add_gtsrb_data(self, transform):
        root = self.args.data
        self.trainset = GTSRB(root, split='train',
                              transform=transform, download=False)
        self.testset = GTSRB(root, split='test',
                             transform=transform, download=False)

        self.trainset.targets = [i[1] for i in self.trainset._samples]
        self.testset.targets = [i[1] for i in self.testset._samples]

        test_data_len = len(self.testset)
        self.ttaset, self.testset = random_split(self.testset, [test_data_len//2, test_data_len-(test_data_len//2)])
        self.ttaset.targets = [i[1] for i in self.ttaset]
        self.testset.targets = [i[1] for i in self.testset]

        self.class_names_list.append(gtsrb_names)
        self.class_names_extended.extend(gtsrb_names)

        self.training_tasks.append(self.trainset)
        self.tta_tasks.append(self.ttaset)
        self.testset_tasks.append(self.testset)
        print("---------------- added GTSRB data --------------")
    
        
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

        self._get_image_list_for_cur_set(task_id, with_buffer=with_buffer)
        idx = copy.deepcopy(self.data_idx)
        curset = self.dataset_collect_fcn(
            self.set, self.class_names_list[task_id], idx, transform=self.transform)

        return curset
    
    def _get_image_list_for_cur_set(self, task, with_buffer=True):
        if f'{task}_{self.mode}' in self.paths.keys():
            # if we have already read paths and labels from fils
            self.data_idx = self.paths[f'{task}_{self.mode}']['data_idx']
        else:
            targets = self.set.targets
            # prepare data idx for current set
            self.data_idx = []
            for idx in range(len(targets)):
                    self.data_idx.append(idx)

            # and save to self.path
            self.paths[f'{task}_{self.mode}'] = {}
            self.paths[f'{task}_{self.mode}']['data_idx'] = self.data_idx

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
        return self.dataset_collect_fcn(self.set, self.class_names_list[task], idx, transform=self.transform)
    
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

