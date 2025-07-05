import os
import wandb
from PIL import Image
from torch import nn,optim
import torchvision
import torch
from torchvision.transforms import ToPILImage

from FeatureExtractor import FeatureExtractor

class Content_loss(nn.Module):
    def __init__(self,target):
        super(Content_loss,self).__init__()
        self.target=target.detach()

    def forward(self,input):
        return nn.functional.mse_loss(input,self.target)

class Style_loss(nn.Module):
    def __init__(self,target):
        super(Style_loss,self).__init__()
        self.target=self.gram_produce(target).detach()

    def gram_produce(self,features):
        b,c,w,h=features.size()
        features=features.view(b*c,w*h)
        gram=torch.mm(features,features.t())

        #归一化
        return gram.div(b*c*w*h)

    def forward(self,features):
        G=self.gram_produce(features)
        return nn.functional.mse_loss(G,self.target)

def load_image(image_path):
    image=Image.open(image_path).convert('RGB')
    compose=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    )
    image=compose(image).unsqueeze(0)
    return image

if __name__=='__main__':
    style_path="style_path/la_muse.jpg"
    content_path="content_path/flower.jpg"
    result_path="result_path/result1.jpg"

    wandb.init(mode="offline",project="wandbuse",name="style-trans-demo")

    num_steps=1000
    count=0
    device=torch.device("cpu")

    vgg19=torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).to(device)
    fx=FeatureExtractor(vgg19).to(device)

    style_image=load_image(style_path)
    style_features=fx(style_image)
    wandb.log({"style_image":wandb.Image(style_path)})

    content_image=load_image(content_path)
    content_features=fx(content_image)
    wandb.log({"content_image": wandb.Image(content_path)})

    input_image=content_image.clone().requires_grad_(True).to(device)
    optimizer=optim.LBFGS([input_image])

    style_loss=[]
    content_loss=[]
    for sl,cl in zip(style_features,content_features):
        style_loss.append(Style_loss(sl))
        content_loss.append(Content_loss(cl))

    style_weight = 1e9
    content_weight = 1e3

    run=[0]
    while run[0]<=num_steps:
        def comstep():
            optimizer.zero_grad()
            input_features=fx(input_image)
            closs=0
            sloss=0

            for cl,infeatures in zip(content_loss,input_features):
                closs+=content_weight*cl(infeatures)
            for sl,infeatures in zip(style_loss,input_features):
                sloss+=style_weight*sl(infeatures)

            loss=closs+sloss
            loss.backward()

            run[0]+=1
            if run[0] % 50 == 0:
                wandb.log({"epoch":run[0],"closs":closs,"sloss":sloss})
                print(
                    f'Step {run[0]}, Content Loss: {closs.item():4f}, Style Loss: {sloss.item():4f}')

            return loss
        optimizer.step(comstep)

    unnormalize = torchvision.transforms.Normalize(
        mean=[-2.118, -2.036, -1.804],
        std=[4.367, 4.464, 4.444]
    )
    result=unnormalize(input_image).to(device)


    to_pil = ToPILImage()
    result = result.squeeze(0)
    result_image = to_pil(result.cpu().clamp(0, 1))
    result_image.save(result_path)
    wandb.log({"result_image": wandb.Image(result_path)})

    wandb.finish()









