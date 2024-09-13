import torch
import numpy as np
import os
import csv
import random
import pandas as pd


# get iid or non-iid data subsets for each client
def allocate_dataset(dataset, client_num, iid: bool, alpha, mode='train'):
    # seed = 190
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    if dataset == 'CREMAD':
        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}
        data_root = ''

        visual_path = r'D:\DATASETS\CREMAD\CREMA-D'
        audio_path = r'D:\DATASETS\CREMAD\CREMA-D\Audio-299\Audio-299'

        train_csv = os.path.join(data_root, dataset + '/train.csv')
        test_csv = os.path.join(data_root, dataset + '/test.csv')

        if mode == 'train':
            csv_file = train_csv
        else:
            csv_file = test_csv

        with open(csv_file, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            subset_class = [[] for _ in range(len(class_dict))]  # instances for each class
            class_sample_num = [0 for _ in range(len(class_dict))]
            for item in csv_reader:
                subset_class[class_dict[item[1]]].append(item[0]+' '+item[1])
                class_sample_num[class_dict[item[1]]] += 1
        print(class_sample_num)

        N = sum(class_sample_num)  # total instance num
        subset_client = [[] for _ in range(client_num)]  # instances for each client

        for k in range(len(class_sample_num)):  # label循环
            k_num = class_sample_num[k]  # the instance number of class k

            proportions = np.random.dirichlet(np.repeat(alpha, client_num))
            proportions = np.array([p*(len(subset_c) < 1.1 * N / client_num) for p, subset_c in zip(proportions, subset_client)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * k_num).astype(int)
            proportions = np.insert(proportions, 0, 0)
            print(proportions)

            random.shuffle(subset_class[k])
            for c in range(client_num):
                subset_client[c] += subset_class[k][proportions[c]:proportions[c+1]]
            # idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(subset_client, np.split())]
        # for cc in subset_client:
        #     print(cc)

        # 保存
        if not os.path.exists('./CREMAD/AllocatedSubset/client-{}-alpha-{}'.format(client_num, alpha)):
            os.makedirs('./CREMAD/AllocatedSubset/client-{}-alpha-{}'.format(client_num, alpha))

        for kk in range(client_num):
            # subset = pd.DataFrame(columns=subset_client[kk])
            # print(subset_client[kk])
            csv_client = open('./CREMAD/AllocatedSubset/client-{}-alpha-{}/client-{}.csv'.format(client_num, alpha, kk), 'w+', newline='', encoding='UTF8')
            writer = csv.writer(csv_client)
            for data in subset_client[kk]:
                # print([data])
                writer.writerow([data])
            csv_client.close()
            # subset.to_csv('./CREMAD/AllocatedSubset/client-{}-alpha-{}/client-{}.csv'.format(client_num, alpha, kk))
    elif dataset == 'AVE':
        data_root = ''
        classes = []

        # visual_path = r'D:\DATASETS\CREMAD\CREMA-D'
        # audio_path = r'D:\DATASETS\CREMAD\CREMA-D\Audio-299\Audio-299'

        train_txt = os.path.join(data_root, dataset + '/trainSet.txt')
        test_txt = os.path.join(data_root, dataset + '/testSet.txt')

        if mode == 'train':
            txt_file = train_txt
        else:
            txt_file = test_txt

        with open(test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        with open(txt_file, encoding='UTF-8-sig') as f:
            files = f.readlines()
            subset_class = [[] for _ in range(len(class_dict))]  # instances for each class
            class_sample_num = [0 for _ in range(len(class_dict))]
            for item in files:
                item = item.split('&')
                subset_class[class_dict[item[0]]].append(item[1] + '/' + item[0])
                class_sample_num[class_dict[item[0]]] += 1
        print(class_sample_num)

        N = sum(class_sample_num)  # total instance num
        subset_client = [[] for _ in range(client_num)]  # instances for each client

        for k in range(len(class_sample_num)):  # label循环
            k_num = class_sample_num[k]  # the instance number of class k

            proportions = np.random.dirichlet(np.repeat(alpha, client_num))
            proportions = np.array(
                [p * (len(subset_c) < 1.1 * N / client_num) for p, subset_c in zip(proportions, subset_client)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * k_num).astype(int)
            proportions = np.insert(proportions, 0, 0)
            print(proportions)

            random.shuffle(subset_class[k])
            for c in range(client_num):
                subset_client[c] += subset_class[k][proportions[c]:proportions[c + 1]]
            # idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(subset_client, np.split())]
        # for cc in subset_client:
        #     print(cc)

        # 保存
        if not os.path.exists('./AVE/AllocatedSubset/client-{}-alpha-{}'.format(client_num, alpha)):
            os.makedirs('./AVE/AllocatedSubset/client-{}-alpha-{}'.format(client_num, alpha))

        for kk in range(client_num):
            # subset = pd.DataFrame(columns=subset_client[kk])
            # print(subset_client[kk])
            csv_client = open('./AVE/AllocatedSubset/client-{}-alpha-{}/client-{}.txt'.format(client_num, alpha, kk),
                              'w+', newline='', encoding='UTF8')
            # writer = csv.writer(csv_client, quoting=csv.QUOTE_NONE, escapechar='\t')
            # writer = csv.writer(csv_client)
            for data in subset_client[kk]:
                # print([data])
                # data = data.strip('\n')
                csv_client.write(data+'\n')
            csv_client.close()
            # subset.to_csv('./CREMAD/AllocatedSubset/client-{}-alpha-{}/client-{}.csv'.format(client_num, alpha, kk))


allocate_dataset(dataset='AVE', client_num=20, iid=False, alpha=3)


