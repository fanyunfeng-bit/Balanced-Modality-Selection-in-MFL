import json
import math
import os
import pickle
import random
from typing import Dict, List, Tuple, Union

from path import Path
from torch.utils.data import Subset, random_split
from torch.utils.data import Dataset
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from transformers import BertTokenizer

from data.CrisisMMD.base_dataset import BaseDataset, expand2square
from data.CrisisMMD.preprocess import clean_text
from termcolor import colored, cprint

dataroot = 'D:\yunfeng\data'

class ModelNet40Dataset(Dataset):

    def __init__(self, args, client_id=None, mode='train', fl=True):
        self.args = args
        self.mode = mode

        self.data_root = './data/'

        self.train_txt = os.path.join(self.data_root, args.dataset + '/train.txt')
        self.test_txt = os.path.join(self.data_root, args.dataset + '/test.txt')

        # class_dict = {}
        # for i, c in enumerate(40):
        #     class_dict[c] = i

        # if mode == 'train':
        #     f = open(self.train_txt, 'r')
        #     self.image = f.readlines()
        # else:
        #     f = open(self.test_txt, 'r')
        #     self.image = f.readlines()

        if mode == 'train':
            if fl:
                if args.alpha >= 1.0:
                    alpha = int(args.alpha)
                else:
                    alpha = args.alpha
                self.data_root = './data/ModelNet40/AllocatedSubset/client-{}-alpha-{}/'.format(args.client_num, alpha)

                self.train_c_txt = os.path.join(self.data_root, 'client-{}.txt'.format(client_id))
                client_file = self.train_c_txt

                f = open(client_file, 'r')
                self.image = f.readlines()

                self.data_num = len(self.image)
        else:
            f = open(self.test_txt, 'r')
            self.image = f.readlines()

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        data_root = 'D:/yunfeng/data/ModelNet40'
        img1_pth = self.image[idx].strip().split('\t')[0]
        img2_pth = self.image[idx].strip().split('\t')[0].replace('v001', 'v007')

        img1_pth = data_root + img1_pth[2:]
        img2_pth = data_root + img2_pth[2:]
        # print('sdasd', self.image[idx].strip().split('\t'))
        label = int(self.image[idx].strip().split('\t')[1])

        img1 = Image.open(img1_pth).convert('RGB')
        img2 = Image.open(img2_pth).convert('RGB')

        img1 = transform(img1).unsqueeze(1)
        img2 = transform(img2).unsqueeze(1)

        #         images = torch.zeros((self.args.num_frame, 3, 224, 224))
        #         for i in range(self.args.num_frame):
        #             #  for i in select_index:
        #             img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
        #             img = transform(img)
        #             images[i] = img

        #         images = torch.permute(images, (1,0,2,3))

        #         # label
        #         label = self.label[idx]

        return img1, img2, label

def get_train_dataset(
        args,
        dataset: str,
        client_id: int,
):
    # 根据已经分配好的数据分布读取数据并构造数据集，在这个函数里不需要分配，只需要根据分配的结果安排就行

    if dataset == 'CREMAD':
        cremad_dataset = CremadDataset(args, client_id, mode='train')
        return cremad_dataset
    elif dataset == 'AVE':
        ave_dataset = AVEDataset(args, client_id, mode='train')
        return ave_dataset
    elif dataset == 'CGMNIST':
        cgmnist_dataset = CGMNISTDataset(args, client_id, mode='train')
        return cgmnist_dataset
    elif dataset == 'CrisisMMD':
        CMMD_dataset = CrisisMMDataset(args, client_id=client_id, phase='train')
        return CMMD_dataset
    elif dataset == 'ModelNet40':
        data = ModelNet40Dataset(args, client_id, mode='train')
        return data


def get_val_dataset(
        args,
        dataset: str
):
    if dataset == 'CREMAD':
        cremad_dataset = CremadDataset(args, mode='val')
        return cremad_dataset
    elif dataset == 'AVE':
        ave_dataset = AVEDataset(args, mode='val')
        return ave_dataset
    elif dataset == 'CGMNIST':
        cgmnist_dataset = CGMNISTDataset(args, mode='val')
        return cgmnist_dataset
    elif dataset == 'CrisisMMD':
        CMMD_dataset = CrisisMMDataset(args, phase='test')
        return CMMD_dataset
    elif dataset == 'ModelNet40':
        data = ModelNet40Dataset(args, mode='val')
        return data


class CremadDataset(Dataset):

    def __init__(self, args, client_id=None, mode='train', fl=True):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        if self.mode == 'train':
            class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

            self.visual_feature_path = r'D:\yunfeng\data\CREMA-D'
            self.audio_feature_path = r'D:\yunfeng\data\CREMA-D\Audio-299'

            if fl:
                if args.alpha >= 1.0:
                    alpha = int(args.alpha)
                else:
                    alpha = args.alpha
                self.data_root = './data/CREMAD/AllocatedSubset/client-{}-alpha-{}/'.format(args.client_num, alpha)

                self.train_csv = os.path.join(self.data_root, 'client-{}.csv'.format(client_id))
                csv_file = self.train_csv

                with open(csv_file, encoding='UTF-8-sig') as f2:
                    csv_reader = csv.reader(f2)
                    for item in csv_reader:
                        it = item[0].split(' ')
                        audio_path = os.path.join(self.audio_feature_path, it[0] + '.pkl')
                        visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps),
                                                   it[0])

                        if os.path.exists(audio_path) and os.path.exists(visual_path):
                            # print('add traing data', visual_path)
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[it[1]])
                        else:
                            continue
                self.data_num = len(self.label)
            else:
                self.train_csv = os.path.join('./data', args.dataset + '/train.csv')
                csv_file = self.train_csv
                with open(csv_file, encoding='UTF-8-sig') as f2:
                    csv_reader = csv.reader(f2)
                    for item in csv_reader:
                        audio_path = os.path.join(self.audio_feature_path, item[0] + '.pkl')
                        visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps),
                                                   item[0])

                        if os.path.exists(audio_path) and os.path.exists(visual_path):
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[item[1]])
                        else:
                            continue

        else:
            self.data_root = './data'
            class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

            self.visual_feature_path = r'D:\yunfeng\data\CREMA-D'
            self.audio_feature_path = r'D:\yunfeng\data\CREMA-D\Audio-299'

            self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

            csv_file = self.test_csv

            with open(csv_file, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)
                for item in csv_reader:
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.pkl')
                    visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps),
                                               item[0])

                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[1]])
                    else:
                        continue
            self.data_num = len(self.label)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # # audio
        # samples, rate = librosa.load(self.audio[idx], sr=22050)
        # resamples = np.tile(samples, 3)[:22050*3]
        # resamples[resamples > 1.] = 1.
        # resamples[resamples < -1.] = -1.
        #
        # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        # spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # #mean = np.mean(spectrogram)
        # #std = np.std(spectrogram)
        # #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            #  for i in select_index:
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.label[idx]

        return spectrogram, images, label


class AVEDataset(Dataset):

    def __init__(self, args, client_id=None, mode='train', fl=True):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []

        self.data_root = './data/'

        self.visual_feature_path = r'D:\yunfeng\data\AVE_Dataset'
        self.audio_feature_path = r'D:\yunfeng\data\AVE_Dataset\Audio-1004-SE'

        self.train_txt = os.path.join(self.data_root, args.dataset + '/trainSet.txt')
        self.test_txt = os.path.join(self.data_root, args.dataset + '/testSet.txt')
        self.val_txt = os.path.join(self.data_root, args.dataset + '/valSet.txt')
        with open(self.test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        if mode == 'train':
            if fl:
                if args.alpha >= 1.0:
                    alpha = int(args.alpha)
                else:
                    alpha = args.alpha
                self.data_root = './data/AVE/AllocatedSubset/client-{}-alpha-{}/'.format(args.client_num, alpha)

                self.train_c_txt = os.path.join(self.data_root, 'client-{}.txt'.format(client_id))
                client_file = self.train_c_txt

                with open(client_file, encoding='UTF-8-sig') as f2:
                    lines = f2.readlines()
                    for line in lines:
                        it = line.strip().split('/')
                        # print(it)
                        audio_path = os.path.join(self.audio_feature_path, it[0] + '.pkl')
                        visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS-SE'.format(self.args.fps),
                                                   it[0])

                        if os.path.exists(audio_path) and os.path.exists(visual_path):
                            # print('add traing data', visual_path)
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[it[1]])
                        else:
                            continue
                self.data_num = len(self.label)
        else:
            txt_file = self.test_txt
            with open(txt_file, 'r') as f2:
                files = f2.readlines()
                for item in files:
                    item = item.split('&')
                    audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                    visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS-SE'.format(self.args.fps), item[1])

                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        if audio_path not in self.audio:
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[item[0]])
                    else:
                        continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # # audio
        # samples, rate = librosa.load(self.audio[idx], sr=22050)
        # resamples = np.tile(samples, 3)[:22050*3]
        # resamples[resamples > 1.] = 1.
        # resamples[resamples < -1.] = -1.
        #
        # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        # spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # #mean = np.mean(spectrogram)
        # #std = np.std(spectrogram)
        # #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        # select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        # select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            # for i, n in enumerate(select_index):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label


class CGMNISTDataset(Dataset):

    def __init__(self, args, client_id=None, mode='train', fl=True):
        self.args = args
        # self.image_gray = []
        # self.image_color = []
        # self.label = []
        self.mode = mode
        # classes = []

        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.gray_dataset = datasets.MNIST(root=r'D:\yunfeng\data\mnist', train=True, download=True,
                                               transform=transform)

        else:
            transform = transforms.Compose([
                # transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.gray_dataset = datasets.MNIST(root=r'D:\yunfeng\data\mnist', train=False, download=True,
                                               transform=transform)

        # colored MNIST
        data_dic = np.load(r'D:\yunfeng\data\colored_mnist\mnist_10color_jitter_var_%.03f.npy' % 0.030,
                           encoding='latin1', allow_pickle=True).item()
        if self.mode == 'train':
            self.colored_image = data_dic['train_image']
            self.colored_label = data_dic['train_label']
        elif self.mode == 'test':
            self.colored_image = data_dic['test_image']
            self.colored_label = data_dic['test_label']

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.ToPIL = transforms.Compose([
            transforms.ToPILImage(),
        ])


    def __len__(self):
        return len(self.gray_dataset)

    def __getitem__(self, idx):
        gray_image, gray_label = self.gray_dataset[idx]

        colored_label = self.colored_label[idx]
        colored_image = self.colored_image[idx]

        colored_image = self.ToPIL(colored_image)

        return gray_image, self.T(colored_image), gray_label, colored_label


class CGMNISTDataset(Dataset):

    def __init__(self, args, client_id=None, mode='train', fl=True):
        self.args = args
        self.mode = mode
        self.fl = fl

        # colored MNIST
        data_dic = np.load(r'D:\yunfeng\data\colored_mnist\mnist_10color_jitter_var_%.03f.npy' % 0.030,
                           encoding='latin1', allow_pickle=True).item()

        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.gray_dataset = datasets.MNIST(root=r'D:\yunfeng\data\mnist', train=True, download=True,
                                               transform=transform)

            self.colored_image = data_dic['train_image']
            self.colored_label = data_dic['train_label']

            if fl:
                if args.alpha >= 1.0:
                    alpha = int(args.alpha)
                else:
                    alpha = args.alpha
                # alpha = 10000000000
                # client_num = 30
                self.data_root = './data/CGMNIST/AllocatedSubset/client-{}-alpha-{}/'.format(args.client_num, alpha)
                # self.data_root = r'D:\MyResearch\Regions\Federated-Learning\Multi-modal-FL\code\Mine\ModalDropFL\data\CGMNIST\AllocatedSubset\client-{}-alpha-{}'.format(client_num, alpha)

                self.train_c_txt = os.path.join(self.data_root, 'client-{}.txt'.format(client_id))
                client_file = self.train_c_txt

                self.sample_index = []
                with open(client_file, encoding='UTF-8-sig') as f2:
                    lines = f2.readlines()
                    for line in lines:
                        it = line.strip()
                        self.sample_index.append(int(it))
                self.data_num = len(self.sample_index)
            # self.fl = False

        else:
            self.fl = False
            transform = transforms.Compose([
                # transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.gray_dataset = datasets.MNIST(root=r'D:\yunfeng\data\mnist', train=False, download=True,
                                               transform=transform)

            self.colored_image = data_dic['test_image']
            self.colored_label = data_dic['test_label']

            self.data_num = len(self.gray_dataset)

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.ToPIL = transforms.Compose([
            transforms.ToPILImage(),
        ])

    def __len__(self):
        if self.fl:
            return len(self.sample_index)
        else:
            return len(self.gray_dataset)

    def __getitem__(self, idx):
        if self.fl:
            gray_image, gray_label = self.gray_dataset[self.sample_index[idx]]
            colored_label = self.colored_label[self.sample_index[idx]]
            colored_image = self.colored_image[self.sample_index[idx]]
        else:
            gray_image, gray_label = self.gray_dataset[idx]
            colored_label = self.colored_label[idx]
            colored_image = self.colored_image[idx]

        colored_image = self.ToPIL(colored_image)

        return gray_image, self.T(colored_image), gray_label

task_dict = {
    'task1': 'informative',
    'task2': 'humanitarian',
    'task2_merged': 'humanitarian',
}

labels_task1 = {
    'informative': 1,
    'not_informative': 0
}

labels_task2 = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 6,
    'missing_or_found_people': 7,
}

labels_task2_merged = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 5,
    'missing_or_found_people': 5,
}



class CrisisMMDataset(BaseDataset):
    def __init__(self, opt, phase='train', cat='all', task='task1', shuffle=False, consistent_only=False, client_id=None):
        super().__init__()
        self.phase = phase
        self.opt = opt
        self.shuffle = shuffle
        self.consistent_only = consistent_only
        self.dataset_root = f'{dataroot}\CrisisMMD_v2.0'
        self.image_root = f'{self.dataset_root}/data_image'
        self.label_map = None
        if task == 'task1':
            self.label_map = labels_task1
        elif task == 'task2':
            self.label_map = labels_task2
        elif task == 'task2_merged':
            self.label_map = labels_task2_merged

        self.client_id = client_id

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        ann_file = '%s/crisismmd_datasplit_all/task_%s_text_img_%s.tsv' % (
            self.dataset_root, task_dict[task], phase
        )

        # Append list of data to self.data_list
        self.read_data(ann_file)

        # if self.shuffle:
        #     np.random.default_rng(seed=0).shuffle(self.data_list)
        self.data_list = self.data_list[:self.opt.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.data_list)), 'yellow')

        self.N = len(self.data_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.transforms = transforms.Compose([
            # transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, Image.BICUBIC)),
            # transforms.Lambda(lambda img: scale_shortside(
            #     img, opt.load_size, opt.crop_size, Image.BICUBIC)),
            transforms.Lambda(lambda img: expand2square(img)),
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomCrop((opt.crop_size, opt.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        if opt.alpha >= 1.0:
            alpha = int(opt.alpha)
        else:
            alpha = opt.alpha
        if phase == 'train':
            self.data_root = './data/CrisisMMD/AllocatedSubset/client-{}-alpha-{}/'.format(opt.client_num, alpha)

            client_file = os.path.join(self.data_root, 'client-{}.txt'.format(self.client_id))

            self.sample_index = []
            with open(client_file, encoding='UTF-8-sig') as f2:
                lines = f2.readlines()
                for line in lines:
                    it = line.strip()
                    self.sample_index.append(int(it))
            self.data_num = len(self.sample_index)

    def read_data(self, ann_file):
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        self.data_list = []

        for l in self.info:
            l = l.rstrip('\n')
            event_name, tweet_id, image_id, tweet_text,	image,	label,	label_text,	label_image, label_text_image = l.split(
                '\t')

            if self.consistent_only and label_text != label_image:
                continue

            self.data_list.append(
                {
                    'path_image': '%s/%s' % (self.dataset_root, image),

                    'text': tweet_text,
                    'text_tokens': self.tokenize(tweet_text),

                    'label_str': label,
                    'label': self.label_map[label],

                    'label_image_str': label_image,
                    'label_image': self.label_map[label_image],

                    'label_text_str': label_text,
                    'label_text': self.label_map[label_text]
                }
            )

    def tokenize(self, sentence):
        ids = self.tokenizer(clean_text(
            sentence), padding='max_length', max_length=40, truncation=True).items()
        return {k: torch.tensor(v) for k, v in ids}

    def __getitem__(self, index):
        if self.phase == 'train':
            data = self.data_list[self.sample_index[index]]
        else:
            data = self.data_list[index]

        to_return = {}
        for k, v in data.items():
            to_return[k] = v

        with Image.open(data['path_image']).convert('RGB') as img:
            image = self.transforms(img)
        to_return['image'] = image
        return to_return

    def __len__(self):
        if self.phase == 'train':
            return len(self.sample_index)
        else:
            return len(self.data_list)

    def name(self):
        return 'CrisisMMDataset'




# args = 0
# data = CGMNISTDataset(args, client_id=0)
# count_class = [0 for _ in range(10)]
# index_class = [[] for _ in range(10)]
# for i, (gray_i, colored_i, gray_l, colored_l) in enumerate(data):
#     print(gray_l, colored_l)
#     count_class[gray_l] += 1
#     index_class[gray_l].append(i)
# print(count_class)
# # print(index_class)
