# Assignment 1 - 35%
In the first practical each group (of 2 students) will choose an image dataset and examine the performance of different settings, such as:

## Requirements
### Comparison of Deep architectures such as AlexNet, VGG, Inception V-3, ResNet, etc.
https://pytorch.org/docs/stable/torchvision/models.html

Importing models 

**Problems with Inception v3**

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


### Using dropout, batch normalization, weight decay, etc.
Batch Normalisation - **implemented - disabled**

Dropout and weight decay (L^2?) - **To be investigated**

### Using different activation functions such as ReLU, ELU, Leaky ReLU, PReLU, SoftPlus, Sigmoid, etc.
TODO: Once settled on architecture, decompose network structure by using print(model). Perhaps, using
model.features.chlildren() -> child params will show the activation functions being used in layer numebrs.
Using this we can enumerate and select activation functions to change. 
```python
TORCH_ENUM_DEFINE(Linear)
TORCH_ENUM_DEFINE(Conv1D)
TORCH_ENUM_DEFINE(Conv2D)
TORCH_ENUM_DEFINE(Conv3D)
TORCH_ENUM_DEFINE(ConvTranspose1D)
TORCH_ENUM_DEFINE(ConvTranspose2D)
TORCH_ENUM_DEFINE(ConvTranspose3D)
TORCH_ENUM_DEFINE(Sigmoid)
TORCH_ENUM_DEFINE(Tanh)
TORCH_ENUM_DEFINE(ReLU)
TORCH_ENUM_DEFINE(LeakyReLU)
TORCH_ENUM_DEFINE(FanIn)
TORCH_ENUM_DEFINE(FanOut)
TORCH_ENUM_DEFINE(Constant)
TORCH_ENUM_DEFINE(Reflect)
TORCH_ENUM_DEFINE(Replicate)
TORCH_ENUM_DEFINE(Circular)
TORCH_ENUM_DEFINE(Nearest)
TORCH_ENUM_DEFINE(Bilinear)
TORCH_ENUM_DEFINE(Bicubic)
TORCH_ENUM_DEFINE(Trilinear)
TORCH_ENUM_DEFINE(Area)
TORCH_ENUM_DEFINE(Sum)
TORCH_ENUM_DEFINE(Mean)
TORCH_ENUM_DEFINE(Max)
TORCH_ENUM_DEFINE(None)
TORCH_ENUM_DEFINE(BatchMean)
TORCH_ENUM_DEFINE(Zeros)
TORCH_ENUM_DEFINE(Border)
TORCH_ENUM_DEFINE(Reflection)
```
```python
'Linear' = {type} <class 'torch.nn.modules.linear.Linear'>
'xavier_uniform_' = {function} <function xavier_uniform_ at 0x7f6438ea8cb0>
'constant_' = {function} <function constant_ at 0x7f6438ea8950>
'xavier_normal_' = {function} <function xavier_normal_ at 0x7f6438ea8d40>
'Parameter' = {type} <class 'torch.nn.parameter.Parameter'>
'Module' = {type} <class 'torch.nn.modules.module.Module'>
'F' = {module} <module 'torch.nn.functional' from '/home/diarmuid/anaconda3/envs/DL-2020/lib/python3.7/site-packages/torch/nn/functional.py'>
'Threshold' = {type} <class 'torch.nn.modules.activation.Threshold'>
'ReLU' = {type} <class 'torch.nn.modules.activation.ReLU'>
'RReLU' = {type} <class 'torch.nn.modules.activation.RReLU'>
'Hardtanh' = {type} <class 'torch.nn.modules.activation.Hardtanh'>
'ReLU6' = {type} <class 'torch.nn.modules.activation.ReLU6'>
'Sigmoid' = {type} <class 'torch.nn.modules.activation.Sigmoid'>
'Tanh' = {type} <class 'torch.nn.modules.activation.Tanh'>
'ELU' = {type} <class 'torch.nn.modules.activation.ELU'>
'CELU' = {type} <class 'torch.nn.modules.activation.CELU'>
'SELU' = {type} <class 'torch.nn.modules.activation.SELU'>
'GLU' = {type} <class 'torch.nn.modules.activation.GLU'>
'Hardshrink' = {type} <class 'torch.nn.modules.activation.Hardshrink'>
'LeakyReLU' = {type} <class 'torch.nn.modules.activation.LeakyReLU'>
'LogSigmoid' = {type} <class 'torch.nn.modules.activation.LogSigmoid'>
'Softplus' = {type} <class 'torch.nn.modules.activation.Softplus'>
'Softshrink' = {type} <class 'torch.nn.modules.activation.Softshrink'>
'MultiheadAttention' = {type} <class 'torch.nn.modules.activation.MultiheadAttention'>
'PReLU' = {type} <class 'torch.nn.modules.activation.PReLU'>
'Softsign' = {type} <class 'torch.nn.modules.activation.Softsign'>
'Tanhshrink' = {type} <class 'torch.nn.modules.activation.Tanhshrink'>
'Softmin' = {type} <class 'torch.nn.modules.activation.Softmin'>
'Softmax' = {type} <class 'torch.nn.modules.activation.Softmax'>
'Softmax2d' = {type} <class 'torch.nn.modules.activation.Softmax2d'>
'LogSoftmax' = {type} <class 'torch.nn.modules.activation.LogSoftmax'>
```
### Using pre-trained networks, data augmentation
Importing Pretrained Models - **Implemented**

Data augmentation using compose.transforms **To be investigated**

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
### Using different optimizers such as SGD, SGD with momentum, Adam, RMSProp, Nadam, etc.
SGD **implemented**

SGD with momentum == 0.009 **implemented**

Investigate the rest

## Assignment Details
Each group can do as many comparison studies as they want and write a report with a short description and
tables / figures with results. The final report should be at most 3 pages in length. It is advised to not use a very large
dataset (e.g. > 200,000 examples), as this would cost too much computational time.

For your report, you need to use the style-file which can be downloaded from: https://bnaic2014.org/?page_id=49
The deadline for mailing your report to m.a.wiering@rug.nl and (a link to) your code is: **28 February 2020, 23.59h**

The report should consist of the following items: Title, student names + numbers, short intro, description of dataset, explanation of which methods you compared, results in a table (it is expected that there will be at least 8 different results), and a short discussion about the results.

___

#### Framework: 
**Pytorch** - Simplicity and models (pretrained available) - Pytorch is the research standard library for
modern deep learning toolbox implementations. 

# Usage

```bash
usage: main.py [-h] [--cuda] [--peregrine-p] [--visual] [--normalise]
               [--batch_size BATCH_SIZE] [--workers WORKERS]
               [--model MODEL_ARCHI] [--train_size TRAINSET_SIZE]

Choose an architecture: | 0 = alexnet || 1 = densenet121 || 2 = densenet161 ||
3 = densenet169 || 4 = densenet201 || 5 = googlenet || 6 = inception_v3 || 7 =
mnasnet0_5 || 8 = mnasnet0_75 || 9 = mnasnet1_0 || 10 = mnasnet1_3 || 11 =
mobilenet_v2 || 12 = resnet101 || 13 = resnet152 || 14 = resnet18 || 15 =
resnet34 || 16 = resnet50 || 17 = resnext101_32x8d || 18 = resnext50_32x4d ||
19 = shufflenet_v2_x0_5 || 20 = shufflenet_v2_x1_0 || 21 = shufflenet_v2_x1_5
|| 22 = shufflenet_v2_x2_0 || 23 = squeezenet1_0 || 24 = squeezenet1_1 || 25 =
vgg11 || 26 = vgg11_bn || 27 = vgg13 || 28 = vgg13_bn || 29 = vgg16 || 30 =
vgg16_bn || 31 = vgg19 || 32 = vgg19_bn || 33 = wide_resnet101_2 || 34 =
wide_resnet50_2 |

optional arguments:
  -h, --help            show this help message and exit
  --cuda, -c            Enable CUDA GPU processing (Default: False)
  --peregrine-p         Set condition for Peregrine Environment (Default:
                        False)
  --visual, -v          Enable matplotlib visuals (Default: False)
  --normalise, -n       Normalise data before training (Default: False)
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch Size (Default: 4)
  --workers WORKERS, -w WORKERS
                        Number of Workers (Default: 2)
  --model MODEL_ARCHI, -m MODEL_ARCHI
                        Model Architecture (Default: 14 = resnet18)
  --train_size TRAINSET_SIZE, -ts TRAINSET_SIZE
                        Set train set size; len(testset) = ts / 5 (Default:
                        20000)

```

### Architecture Codes:
* 0 = alexnet
* 1 = densenet121
* 2 = densenet161
* 3 = densenet169
* 4 = densenet201
* 5 = googlenet
* 6 = inception_v3
* 7 = mnasnet0_5
* 8 = mnasnet0_75
* 9 = mnasnet1_0
* 10 = mnasnet1_3
* 11 = mobilenet_v2
* 12 = resnet101
* 13 = resnet152
* 14 = resnet18
* 15 = resnet34
* 16 = resnet50 
* 17 = resnext101_32x8d
* 18 = resnext50_32x4d
* 19 = shufflenet_v2_x0_5
* 20 = shufflenet_v2_x1_0
* 21 = shufflenet_v2_x1_5
* 22 = shufflenet_v2_x2_0
* 23 = squeezenet1_0
* 24 = squeezenet1_1
* 25 = vgg11
* 26 = vgg11_bn
* 27 = vgg13
* 28 = vgg13_bn
* 29 = vgg16
* 30 = vgg16_bn
* 31 = vgg19
* 32 = vgg19_bn
* 33 = wide_resnet101_2
* 34 = wide_resnet50_2
