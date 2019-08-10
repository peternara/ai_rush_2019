import os
import numpy as np
import pathlib
import pandas as pd
import pickle as pkl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nsml import DATASET_PATH
#from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
#from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import PIL

def data_generator_for_keras(input_size=128,
                    batch_size=64,
                    num_workers=0,
                    ):
    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
    train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    train_meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)
    print("total train meta data shape : ", train_meta_data.shape)
    print(train_meta_data.head(10))
    
#    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=777)
    train_df, valid_df  = train_test_split(train_meta_data, test_size=0.1, random_state=777)
    print("train data shape : ", train_df.shape)
    print(train_df.head(5))
    print("valid data shape : ", valid_df.shape)
    print(valid_df.head(5))

def train_dataloader(input_size=128,
                    batch_size=64,
                    num_workers=0,
                    use_only_single = False, test_bs = False, br_multi_oh= False, print_nor_info = False):
    
    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
    train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    train_meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)
    print("total train meta data shape : ", train_meta_data.shape)#(581777, 3)
    print(train_meta_data.head(10))

    label_matrix = np.load(train_label_path)
    print("label_matrix shape : ", label_matrix.shape)
    class_count = label_matrix.sum(axis=0)
    print('class_count only disp top 10 : ',class_count[:10])
    single_list = []
    for i in range(label_matrix.shape[0]):
        if label_matrix[i,:].sum() == 1:
            single_list.append(i)
    print('single_list count',len(single_list))

    #print('single_list',single_list)single_list count 205173
    # mean tensor([0.8674, 0.8422, 0.8218]) std tensor([0.2407, 0.2601, 0.2791])
    #mean_v = [0.8674, 0.8422, 0.8218]
    #std_v = [0.2407, 0.2601, 0.2791]

    data_seed = 777
    if use_only_single == True:
        train_meta_data = train_meta_data.loc[single_list]
        label_matrix_df = pd.DataFrame(label_matrix)
        label_single_df = label_matrix_df.loc[single_list]
        label_matrix = label_single_df.to_numpy()
        print("single total train meta data shape : ", train_meta_data.shape)
        print("single label_matrix shape : ", label_matrix.shape)
        print('single_label distribut:',label_matrix.sum(axis=0))
        print('single_label distribut min :',label_matrix.sum(axis=0).min())

        #sf = np.argmax(label_matrix,axis=1)
        #single_data_df, single_label
        train_df, valid_df, train_label, valid_label  = train_test_split(train_meta_data, label_matrix
                                                                     , test_size=0.05, random_state=data_seed) #, stratify=sf ) #this data isn't single image case. 
    else:
        # multi but .. stratify iam lazy TT
        #sf = np.argmax(label_matrix,axis=1)
        train_df, valid_df, train_label, valid_label  = train_test_split(train_meta_data, label_matrix
                                                                     , test_size=0.05, random_state=data_seed) 
    

    print("train data shape : ", train_df.shape)
    print(train_df.head(5))
    print("valid data shape : ", valid_df.shape)
    print(valid_df.head(5))
    print("train_label shape : ", train_label.shape)
    print("valid_label shape : ", valid_label.shape)


    print("batch_size : ", batch_size)

    if test_bs ==True:
        train_df = train_df.head(batch_size)
        valid_df = valid_df.head(batch_size)
        train_label = train_label[:batch_size]
        valid_label = valid_label[:batch_size]


    dataloader = DataLoader(
        AIRushDataset(image_dir, train_df, label=train_label, br_multi_oh =br_multi_oh,
                      transform=transforms.Compose([transforms.Resize((int(input_size*1.1),int( input_size*1.1)))
                                                    ,transforms.RandomCrop((input_size, input_size), padding_mode='symmetric')
                                                    , transforms.RandomHorizontalFlip()
                                                    ,transforms.RandomRotation(20, resample=PIL.Image.BILINEAR)
                                                    , transforms.ToTensor()
                                                    #,transforms.Normalize(mean=mean_v, std=std_v)
                                                    ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    valid_dataloader = DataLoader(
        AIRushDataset(image_dir, valid_df, label=valid_label, 
                      transform=transforms.Compose([transforms.Resize((input_size, input_size))
                                                    , transforms.ToTensor()
                                                    #,transforms.Normalize(mean=mean_v, std=std_v)
                                                    ])),
        batch_size=batch_size//10,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    return dataloader, valid_dataloader


class AIRushDataset(Dataset):
    def __init__(self, image_data_path, meta_data,label=None, transform=None, use_multi=False, br_multi_oh=False):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label = label
        self.transform = transform
        self.use_multi = use_multi
        self.br_multi_oh = br_multi_oh
        if self.label is not None:
            self.label_matrix = label

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir , str(self.meta_data['package_id'].iloc[idx]) , str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load() # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3]) # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)

        #print('new_img.mean',new_img.mean(),'new_img.std', new_img.std())
        
        if self.label is not None:
            if self.br_multi_oh == True:
                tags_single = torch.tensor(np.argmax(self.label_matrix[idx])) 
                tags_multi = torch.tensor((self.label_matrix[idx]))
                tagss = []
                tagss.append(tags_single)
                tagss.append(tags_multi)
                return new_img, tagss         

            if self.use_multi == True:
                tags = torch.tensor((self.label_matrix[idx]))
            else :
                tags = torch.tensor(np.argmax(self.label_matrix[idx])) # here, we will use only one label among multiple labels.
            return new_img, tags
        else:
            return new_img







    #if print_nor_info == True: # mean tensor([0.8674, 0.8422, 0.8218]) std tensor([0.2407, 0.2601, 0.2791])
    #    cal_dataloader = DataLoader(
    #        AIRushDataset(image_dir, train_meta_data, label=None, 
    #                      transform=transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
    #        batch_size=batch_size,
    #        shuffle=False,
    #        num_workers=1)
    #    mean = 0.
    #    std = 0.
    #    nb_samples = 0.
    #    for data in cal_dataloader:
    #        batch_samples = data.shape[0]#(0)
    #        data = data.view(batch_samples, data.size(1), -1)
    #        mean += data.mean(2).sum(0)
    #        std += data.std(2).sum(0)
    #        nb_samples += batch_samples
    #        #print('process imgs', nb_samples)

    #    mean /= nb_samples
    #    std /= nb_samples
    #    print('mean',mean, 'std', std)