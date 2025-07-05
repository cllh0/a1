import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.models import VGG19_Weights
from tqdm import trange

import utilis
from Net import Net
from torch import nn,optim
from utilis import StyleLoader
from FeatureExtractor import FeatureExtractor
import wandb


if __name__=='__main__':
    path="Dataset/content"
    style_path="Dataset/style"
    wandb.init(mode="offline", project="wandbuse", name="style-trans-demo")

    transform=transforms.Compose([transforms.Resize(244),
                                  transforms.CenterCrop(244),
                                  transforms.ToTensor(),
                                  transforms.Lambda(lambda x:x.mul(255))
                                  ])
    train_dataset=datasets.ImageFolder(path,transform=transform)
    train_loader=DataLoader(train_dataset,batch_size=4)

    device=torch.device("cpu")

    style_model=Net(64).to(device)
    optimizer=optim.Adam(style_model.parameters(),lr=1e-3)
    mse_loss=torch.nn.MSELoss()
    vgg=torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).to(device).eval()
    fx=FeatureExtractor(vgg)

    style_loader=utilis.StyleLoader(style_path,244)


    epochs=10
    tbar=trange(epochs)
    content_weight=1
    style_weight=5
    for e in tbar:
        style_model.train()
        content_loss=0
        style_loss=0
        count=0
        for batch_id,(x,_) in enumerate(train_loader):
            n_batch=len(x)
            count+=n_batch
            optimizer.zero_grad()
            x = Variable(utilis.preprocess_batch(x))

            style_v=style_loader.get(batch_id)
            style_model.setTarget(style_v)
            style_v=utilis.subtract_imagenet_mean_batch(style_v)

            style_features=fx(style_v)
            gram_style=[utilis.get_matrix(y) for y in style_features]

            y=style_model(x)   #取得内容图片的风格迁移
            xc=Variable(x.data.clone())

            y=utilis.subtract_imagenet_mean_batch(y)
            xc=utilis.subtract_imagenet_mean_batch(xc)

            features_y=fx(y)
            features_xc=fx(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss=0
            for m in range(len(features_y)):
                gram_y=utilis.get_matrix(features_y[m])
                gram_s=Variable(gram_style[m].data,requires_grad=False).repeat(n_batch,1,1)
                style_loss+=style_weight*mse_loss(gram_y,gram_s[:n_batch,:,:])

            total_loss=content_loss+style_loss
            total_loss.backward()
            optimizer.step()

        fixed_batch,_ = next(iter(train_loader))
        fixed_batch=fixed_batch.to(device)

        style_model.eval()

        for idx,content_image in enumerate(fixed_batch):
            content_image=content_image.unsqueeze(0).to(device)
            content_image = Variable(utilis.preprocess_batch(content_image))
            example_style=style_loader.get(0).to(device)
            style_model.setTarget(example_style)
            output=style_model(content_image)
            filename = os.path.join("result_path", "{}_.png".format(idx+5))
            utilis.tensor_save_bgrimage(output.data[0], filename)
            print(filename)
            wandb.log({"result_image": wandb.Image(filename)})