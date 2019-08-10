import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Baseline, Resnet
import nsml
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import train_dataloader
from dataloader import AIRushDataset
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from torchsummary import summary


# hyper param setting
use_train_time_multi_calss_info_add = True  # default is True
test_bs = False # search batch size when debugging , default is False
use_pretrained = True # test model, no download , default is True

re_train_info = {'session':'team_27/airush1/281', 'checkpoint':'9'}  #= None
# {'session':'team_27/airush1/262', 'checkpoint':'1'} # = None   #se_resnext50_32x4d


if re_train_info is not None:
    use_pretrained = False

down_lr_step = 3
start_lr = 0.001

pre_trained_model_list = [{'model':'se_resnext50_32x4d', 'batch_size':160}  #0
                          ,{'model':'inceptionresnetv2', 'batch_size':130} #1
                          ,{'model':'nasnetamobile', 'batch_size':200} #2
                          ,{'model':'efficientnet-b4', 'batch_size':70}] 
select_model_num = 3
_BATCH_SIZE = pre_trained_model_list[select_model_num]['batch_size']
model_name = pre_trained_model_list[select_model_num]['model']


mean_v = [0.8674, 0.8422, 0.8218]
std_v = [0.2407, 0.2601, 0.2791]


def to_np(t):
    return t.cpu().detach().numpy()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def bind_model(model_nsml):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
                    'model': model_nsml.state_dict(),
                }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(save_state_path)
        model_nsml.load_state_dict(state['model'])
        
    def infer(test_image_data_path, test_meta_data_path):
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)
        
        input_size=224 # you can change this according to your model.
        batch_size=_BATCH_SIZE//4 # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        # test time gpu memory error, so i reduce batch size when test time
        device = 0
        
        dataloader = DataLoader(
                        AIRushDataset(test_image_data_path, test_meta_data, label=None,
                                      transform=transforms.Compose([transforms.Resize((input_size, input_size))
                                                                    , transforms.ToTensor()
                                                                    #,transforms.Normalize(mean=mean_v, std=std_v)
                                                                    ])),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)
        
        model_nsml.to(device)
        model_nsml.eval()
        predict_list = []
        for batch_idx, image in enumerate(dataloader):
            image = image.to(device)
            image_lr = image.flip(3) #batch, ch, h, w

            output_org = model_nsml(image).double()
            output_lr = model_nsml(image_lr).double()
            output = output_org + output_lr

            output_prob = F.softmax(output, dim=1)
            predict = np.argmax(to_np(output_prob), axis=1)

            predict_list.append(predict)
                
        predict_vector = np.concatenate(predict_list, axis=0)
        return predict_vector # this return type should be a numpy array which has shape of (138343, 1)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    
    # custom args
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=_BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--output_size', type=int, default=350) # Fixed
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=start_lr)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device


    if test_bs == True:
        use_pretrained = False

    if model_name.split('-')[0]=='efficientnet':
        if use_pretrained ==True:
            model = EfficientNet.from_pretrained(model_name, num_classes=args.output_size)
        else:
            model = model = EfficientNet.from_name(model_name, override_params={'num_classes': args.output_size})
    else:
        model = make_model(model_name, num_classes=args.output_size, pretrained=use_pretrained, pool=nn.AdaptiveAvgPool2d(1))


    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    criterion = nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()#CrossEntropyLoss() #multi-class classification task
    criterion1 = nn.BCEWithLogitsLoss()

    model = model.to(device) #use gpu
    #summary(model, (3,args.input_size,args.input_size))
    # DONOTCHANGE: They are reserved for nsml
    bind_model(model)

    if re_train_info is not None and args.mode == "train":
        print(re_train_info)
        nsml.load(checkpoint=re_train_info['checkpoint'], session=re_train_info['session']) 
        nsml.save('dontgiveup')


    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == "train":
        # Warning: Do not load data before this line
        dataloader, valid_dataloader = train_dataloader(args.input_size, args.batch_size, args.num_workers, test_bs =  test_bs
                                                        , br_multi_oh=use_train_time_multi_calss_info_add#)
                                                        ,print_nor_info = False)


        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,	
                              step_size=down_lr_step,	
                              gamma=0.5)

        for epoch_idx in range(1, args.epochs + 1):
            total_loss = 0
            total_correct = 0
            total_valid_loss = 0
            total_valid_correct = 0
            scheduler.step()
            print('Epoch:', epoch_idx,'LR:', scheduler.get_lr())
            model.train()
            for batch_idx, (image, tags) in enumerate(dataloader):
                optimizer.zero_grad()
                #print('image.shape',image.shape,'tags.shape',tags.shape)
                image = image.to(device)                
                if use_train_time_multi_calss_info_add ==True:
                    tags_m = tags[1].to(device)
                    tags = tags[0].to(device)
                else:
                    tags = tags.to(device)


                output = model(image).double()
                #print('output.shape',output.shape)
                loss = criterion(output, tags) + criterion1(output, tags_m) #train time support
                loss.backward()
                optimizer.step()

                output_prob = F.softmax(output, dim=1)
                predict_vector = np.argmax(to_np(output_prob), axis=1)
                label_vector = to_np(tags)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)

                if batch_idx % args.log_interval == 0:
                    print('Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(batch_idx,
                                                                             len(dataloader),
                                                                             loss.item(),
                                                                             accuracy))
                total_loss += loss.item()
                total_correct += bool_vector.sum()

            ## validation
            model.eval()
            for batch_idx, (image, tags) in enumerate(valid_dataloader):
                image = image.to(device)

                tags = tags.to(device)
                output = model(image).double()
                loss = criterion(output, tags)
                output_prob = F.softmax(output, dim=1)
                predict_vector = np.argmax(to_np(output_prob), axis=1)
                label_vector = to_np(tags)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)
                total_valid_loss += loss.item()
                total_valid_correct += bool_vector.sum()


                    
            nsml.save(epoch_idx)
            print('Epoch {} / {}: Loss {:2.4f} / Acc {:2.4f}, Valid_Loss {:2.4f} / Valid_Acc {:2.4f}'.format(epoch_idx,
                                                           args.epochs,
                                                           total_loss/len(dataloader.dataset),
                                                           total_correct/len(dataloader.dataset),
                                                           total_valid_loss/len(valid_dataloader.dataset),
                                                           total_valid_correct/len(valid_dataloader.dataset)))
            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                "train__Loss": total_loss/len(dataloader.dataset),
                "train__Accuracy": total_correct/len(dataloader.dataset),
                "valid__Loss": total_valid_loss/len(valid_dataloader.dataset),
                "valid__Accuracy": total_valid_correct/len(valid_dataloader.dataset),
                })
