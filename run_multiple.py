import os
import torch
import torchvision
from PIL import Image
from PIL.features import check_feature
import wandb

from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torchvision.models import VGG19_Weights
from torchvision.transforms import ToPILImage

from FeatureExtractor import FeatureExtractor


class MyData(Dataset):
    def __init__(self,root_dir,label_dir,transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, index):
        img_name=self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        if self.transform:
            img = self.transform(img)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

class ContentLoss(nn.Module):
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        self.target=target.detach()

    def forward(self,input):
        return nn.functional.mse_loss(input,self.target)

class StyleLoss(nn.Module):
    def __init__(self,target):
        super(StyleLoss,self).__init__()
        self.target=self.getgram(target).detach()

    def getgram(self,features):
        b,c,h,w=features.size()
        features=features.view(b*c,h*w)
        gram=torch.mm(features,features.t())
        return gram.div(b*c*h*w)

    def forward(self,input):
        G=self.getgram(input)
        return nn.functional.mse_loss(G,self.target)

if __name__ == '__main__':
    path='Dataset/appleorange_data'
    lableA='apple'
    lableB='orange'
    result_path='Dataset/appleorange_result'

    wandb.init(mode="offline", project="wandbuse", name="style-trans-runmul-demo")

    numsteps=100
    device=torch.device("cpu")
    style_weight = 1e9
    content_weight = 1e3

    vgg19=torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).to(device)
    fx=FeatureExtractor(vgg19).to(device)

    compose=torchvision.transforms.Compose([
        torchvision.transforms.Resize((244,244)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    datasetA=MyData(path,lableA,transform=compose)
    dataloaderA=DataLoader(datasetA,batch_size=16)
    datasetB = MyData(path, lableB, transform=compose)
    dataloaderB = DataLoader(datasetB, batch_size=16)
    count=0

    for dataA,dataB in zip(dataloaderA,dataloaderB):
        style_batch,style_target=dataA  #第一个为一批次图像，第二个为标签，Apple
        content_batch,content_target=dataB
        style_batch=style_batch.to(device)
        content_batch=content_batch.to(device)

        for s_image,c_image in zip(style_batch,content_batch):

            s_image=s_image.unsqueeze(0).to(device)
            c_image=c_image.unsqueeze(0).to(device)

            s_features=fx(s_image)
            c_features=fx(c_image)

            input_image=c_image.clone().requires_grad_(True).to(device)
            optimizer=optim.LBFGS([input_image])

            s_loss=[]
            c_loss=[]
            for sl,cl in zip(s_features,c_features):
                s_loss.append(StyleLoss(sl))
                c_loss.append(ContentLoss(cl))

            run=[0]
            while run[0]<=numsteps:
                def com():
                    optimizer.zero_grad()
                    input_features=fx(input_image)
                    clossnum=0
                    slossnum=0
                    for cl,input_f in zip(c_loss,input_features):
                        clossnum+=content_weight*cl(input_f)
                    for sl,input_f in zip(s_loss,input_features):
                        slossnum+=style_weight*sl(input_f)
                    loss=clossnum+slossnum
                    loss.backward()

                    run[0] += 1
                    if run[0] % 50 == 0:
                        wandb.log({"epoch": run[0], "closs": clossnum, "sloss": slossnum})
                        print(
                            f'Step {run[0]}, Content Loss: {clossnum.item():4f}, Style Loss: {slossnum.item():4f}')

                    return loss
                optimizer.step(com)

            unnormalize = torchvision.transforms.Normalize(
                mean=[-2.118, -2.036, -1.804],
                std=[4.367, 4.464, 4.444]
            )
            result = unnormalize(input_image).to(device)
            to_pil = ToPILImage()
            result = result.squeeze(0)
            result_image = to_pil(result.cpu().clamp(0, 1))
            final_path=result_path+f"pic{count}.jpg"
            result_image.save(final_path)
            wandb.log({"result_image": wandb.Image(final_path)})
            count+=1

    wandb.finish()


















