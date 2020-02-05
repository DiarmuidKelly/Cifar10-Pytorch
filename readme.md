# Assignment 1 - 35%
In the first practical each group (of 2 students) will choose an image dataset and examine the performance of different settings, such as:

## Requirements
### Comparison of Deep architectures such as AlexNet, VGG, Inception V-3, ResNet, etc.
https://pytorch.org/docs/stable/torchvision/models.html

Importing models 


```python
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
shufflenet = models.shufflenet_v2_x1_0()
mobilenet = models.mobilenet_v2()
resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
mnasnet = models.mnasnet1_0()
```

Importing Pretrained Models

```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
```
### Using dropout, batch normalization, weight decay, etc.

### Using different activation functions such as ReLU, ELU, Leaky ReLU, PReLU, SoftPlus, Sigmoid, etc.

### Using pre-trained networks, data augmentation

### Using different optimizers such as SGD, SGD with momentum, Adam, RMSProp, Nadam, etc.


Each group can do as many comparison studies as they want and write a report with a short description and
tables / figures with results. The final report should be at most 3 pages in length. It is advised to not use a very large
dataset (e.g. > 200,000 examples), as this would cost too much computational time.

For your report, you need to use the style-file which can be downloaded from: https://bnaic2014.org/?page_id=49
The deadline for mailing your report to m.a.wiering@rug.nl and (a link to) your code is: **28 February 2020, 23.59h**

The report should consist of the following items: Title, student names + numbers, short intro, description of dataset, explanation of which methods you compared, results in a table (it is expected that there will be at least 8 different results), and a short discussion about the results.

___

#### Proposed Framework: 
**Pytorch** - Simplicity and models (pretrained available) - Pytorch is the research standard library for
modern deep learning toolbox implementations. 