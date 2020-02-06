import torchvision.models as models
import torch.nn as nn

def test():
    resnet18 = models.resnet18()
    print(resnet18)
    resnet18.layer1[0].relu = nn.LeakyReLU(inplace=True)
    resnet18.layer1[1].relu = nn.LeakyReLU(inplace=True)
    print(resnet18)



if __name__ == "__main__":
    test()
