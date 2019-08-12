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
from dataloader import AIRushDataset, ObjROI
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
import PIL

_BATCH_SIZE = 24

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, Ar=0.5, Br=0.5):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.Ar = Ar
        self.Br = Br
        
    def forward(self, x):
        xa = self.modelA(x)
        xb = self.modelB(x)
        return xa*self.Ar+xb*self.Br

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
        batch_size=_BATCH_SIZE//2 # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        # test time gpu memory error, so i reduce batch size when test time
        device = 0
        
        dataloader = DataLoader(
                        AIRushDataset(test_image_data_path, test_meta_data, label=None,
                                      transform=transforms.Compose([ObjROI()
                                          ,transforms.Resize((input_size, input_size))
                                                                    , transforms.ToTensor()
                                                                    #,transforms.Normalize(mean=mean_v, std=std_v)
                                                                    ])),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=3,
                        pin_memory=True)
        
        model_nsml.to(device)
        model_nsml.eval()
        predict_list = []
        for batch_idx, image in enumerate(dataloader):
            image = image.to(device)
            image_lr = image.flip(3) #batch, ch, h, w

            cat_tensor = torch.cat((image, image_lr))
            output_cat = model_nsml(cat_tensor).double()
            
            output_org = output_cat[:batch_size,:]
            ouput_lr = output_cat[batch_size:,:]
            #output_org = model_nsml(image).double()
            #output_lr = model_nsml(image_lr).double()
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
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()




    torch.manual_seed(args.seed)
    device = args.device


    use_ensemble_model_session = [{'session':'team_27/airush1/372', 'checkpoint':'0'} #'efficientnet-b4'
                                  , {'session':'team_27/airush1/354', 'checkpoint':'5'}] #se_resnext50_32x4d

    modelA = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': args.output_size})
    modelB = make_model('se_resnext50_32x4d', num_classes=args.output_size, pretrained=False, pool=nn.AdaptiveAvgPool2d(1))
    
    # DONOTCHANGE: They are reserved for nsml
    bind_model(modelA)
    re_train_info = use_ensemble_model_session[0]#'efficientnet-b4'
    nsml.load(checkpoint=re_train_info['checkpoint'], session=re_train_info['session']) 
    bind_model(modelB)
    re_train_info = use_ensemble_model_session[1]#'se_resnext50_32x4d'
    nsml.load(checkpoint=re_train_info['checkpoint'], session=re_train_info['session']) 

    model = MyEnsemble(modelA,modelB)

    model = model.to(device) #use gpu
    #summary(model, (3,args.input_size,args.input_size))
    # DONOTCHANGE: They are reserved for nsml
    bind_model(model)
    nsml.save('dontgiveup')


    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == "train":
        # Warning: Do not load data before this line
        dataloader, valid_dataloader = train_dataloader(args.input_size, args.batch_size*5, args.num_workers, test_bs =  False
                                                        , br_multi_oh=True#)
                                                        ,print_nor_info = False,use_last_fine_tune=True)

        def validation(val_step_num):
            total_valid_correct = 0
            model.eval()
            for batch_idx, (image, tags) in enumerate(valid_dataloader):
                image = image.to(device)

                image_lr = image.flip(3) #batch, ch, h, w
                cat_tensor = torch.cat((image, image_lr))
                output_cat = model_nsml(cat_tensor).double()
            
                output_org = output_cat[:batch_size,:]
                ouput_lr = output_cat[batch_size:,:]
                #output_org = model_nsml(image).double()
                #output_lr = model_nsml(image_lr).double()
                output = output_org + output_lr

                tags = tags.to(device)
                #output = model(image).double()
                output_prob = F.softmax(output, dim=1)
                predict_vector = np.argmax(to_np(output_prob), axis=1)
                label_vector = to_np(tags)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)
                total_valid_correct += bool_vector.sum()
            nsml.save(val_step_num)
            print('val_step_num {} : Valid_Acc {:2.4f}'.format(val_step_num,
                                                           total_valid_correct/len(valid_dataloader.dataset)))
            nsml.report(
                summary=True,
                step=val_step_num,
                scope=locals(),
                **{
                "valid__Accuracy": total_valid_correct/len(valid_dataloader.dataset),
                })


        for i in range(10):
            pr = np.random.uniform(-0.2, 0.2)
            Ar = 0.5 + pr
            Br = 1 - Ar
            print('Ar',Ar,'Br',Br)
            model = MyEnsemble(modelA,modelB,Ar, Br)
            validation(i)