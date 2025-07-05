import os
from modulefinder import Module

import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.nn import Conv2d
from torch_geometric.nn import InstanceNorm
import numpy as np

class GramMatrix(nn.Module):
    def forward(self,y):
        (b,c,w,h)=y.size()
        features=y.view(b,c,w*h)
        feature_t=features.transpose(1,2)
        gram=features.bmm(feature_t)/(c*w*h)
        return gram

class ConvLayer(nn.Module):
    def __init__(self,in_put,out_put,stride,kernel_size):
        super(ConvLayer,self).__init__()
        reflect_padding=int(np.floor(kernel_size/2))
        self.reflect_pad=nn.ReflectionPad2d(reflect_padding)
        self.conv2d=nn.Conv2d(in_put,out_put,kernel_size,stride)

    def forward(self,x):
        out=self.reflect_pad(x)
        out=self.conv2d(out)
        return out

class Bottleneck(nn.Module):  #下采样
    def __init__(self,in_put,out_put,stride=1,downsample=0,norm_layer=nn.BatchNorm2d):
        super(Bottleneck,self).__init__()
        self.expansion=4
        self.downsample=downsample

        if self.downsample!=0:
            self.residual=Conv2d(in_put,out_put*self.expansion,kernel_size=1,stride=stride)

        conv_block=[]
        conv_block+=[norm_layer(in_put),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_put,out_put,stride=1,kernel_size=1)]

        conv_block+=[norm_layer(out_put),
                     nn.ReLU(inplace=True),
                     ConvLayer(out_put,out_put,stride=stride,kernel_size=3)]

        conv_block+=[norm_layer(out_put),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(out_put,out_put*self.expansion,stride=1,kernel_size=1)]
        self.conv_block=nn.Sequential(*conv_block)

    def forward(self,x):
        if self.downsample is not None:
            residual=self.residual(x)
        else:
            residual=x
        return residual+self.conv_block(x)

class Inspiration(nn.Module):
    def __init__(self,C,B=1):
        super(Inspiration,self).__init__()
        self.weight=nn.Parameter(torch.Tensor(1,C,C),requires_grad=True)
        self.G=Variable(torch.Tensor(B,C,C),requires_grad=True)
        self.C=C
        self.reset_parameter()

    def reset_parameter(self):
        self.weight.data.uniform_(0.0,2.0)

    def setTarget(self,target):
        self.G=target

    def forward(self,X):
        self.P=torch.bmm(self.weight.expand_as(self.G),self.G)
        return torch.bmm(self.P.transpose(1,2).expand(X.size(0), self.C, self.C),
                         X.view(X.size(0),X.size(1),-1)).view_as(X)

class UpConvLayer(nn.Module):
    def __init__(self,in_put,out_put,kernel_size,stride,upsample=None):
        super(UpConvLayer,self).__init__()
        self.upsample=upsample
        if self.upsample is not None:
            self.upsample_layer=nn.Upsample(scale_factor=upsample)
        self.reflect_padding=int(np.floor(kernel_size/2))
        if self.reflect_padding!=0:
            self.reflect_pad=nn.ReflectionPad2d(self.reflect_padding)
        self.conv2d=nn.Conv2d(in_put,out_put,kernel_size,stride)

    def forward(self,x):
        if self.upsample:
            x=self.upsample_layer(x)
        if self.reflect_padding!=0:
            x=self.reflect_pad(x)
        out=self.conv2d(x)
        return out

class UpBottleNeck(nn.Module):
    def __init__(self,input,output,stride=2,norm_layer=nn.BatchNorm2d):
        super(UpBottleNeck,self).__init__()
        self.expansion=4
        self.residual_layer=UpConvLayer(input,output*self.expansion,kernel_size=1,stride=1,
                                        upsample=stride)

        conv_block = []
        conv_block += [norm_layer(input),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(input, output, kernel_size=1, stride=1)]
        conv_block += [norm_layer(output),
                       nn.ReLU(inplace=True),
                       UpConvLayer(output, output, kernel_size=3, stride=1, upsample=stride)]
        conv_block += [norm_layer(output),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(output, output * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self,x):
        return self.residual_layer(x)+self.conv_block(x)

class Net(nn.Module):
    def __init__(self,in_put=3,out_put=3,nfg=64,norm_layer=nn.InstanceNorm2d,n_blocks=6):
        super(Net,self).__init__()
        self.gram=GramMatrix()

        block=Bottleneck
        upblock=UpBottleNeck
        expansion=4

        model1=[]
        model1+=[ConvLayer(3,64,kernel_size=7,stride=1),
                 norm_layer(64),
                 nn.ReLU(inplace=True),
                 block(64,32,2,downsample=1,norm_layer=norm_layer),
                 block(32*expansion,nfg,2,1,norm_layer)
                 ]
        self.model1=nn.Sequential(*model1)

        model=[]
        self.ins=Inspiration(nfg*expansion)
        model+=[self.model1]
        model+=[self.ins]

        for i in range(n_blocks):
            model+=[block(nfg*expansion,nfg,1,None,norm_layer)]

        model+=[upblock(nfg*expansion,32,2,norm_layer),
                upblock(32*expansion,16,2,norm_layer),
                norm_layer(16*expansion),
                nn.ReLU(inplace=True),
                ConvLayer(16*expansion,out_put,kernel_size=7,stride=1)
                ]

        self.model=nn.Sequential(*model)

    def setTarget(self, Xs):
        F = self.model1(Xs)
        G = self.gram(F)
        self.ins.setTarget(G)

    def forward(self, input):
        return self.model(input)




