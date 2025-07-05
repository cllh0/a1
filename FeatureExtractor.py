from torch import nn
class FeatureExtractor(nn.Module):
    def __init__(self,model):
        super(FeatureExtractor,self).__init__()
        self.model=model
        self.feature=self.model.features[:21].eval()

    def forward(self,x):
        features=[]
        for i,layer in enumerate(self.feature):
            x=layer(x)
            if i in {0, 5, 10, 19, 21}:
                features.append(x)
        return features