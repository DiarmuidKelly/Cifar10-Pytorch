import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import time
import argparse

freeze_layers = False
dropout = False
batch_size = 4
workers = 2
normalise = False
include_visuals = False
use_cuda = False
peregrine = False
load_from_memory = False
pretrain = False
optimizer_choice = 1
epochs = 2
learning_rate = 0
momentum = 0
weight_decay = 0
trainset_size = 20000  # TS should be lower than 12500
testset_size = trainset_size / 5

# Constants
mini_batch_print = 500
model_names = []

student_number = "s4091221"  # Used for peregrine directory


def test(network_architecture):

    ###########################################################################################

    if normalise:
        transform = transforms.Compose(
            transforms.Resize(256),  # Should resizing do anything to the outcome?
            transforms.CenterCrop(224),  # Why 224?
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),  # Should resizing do anything to the outcome?
            transforms.CenterCrop(224),
            transforms.ToTensor()])  # TODO: transforms on inputs depedent on architecture

    if peregrine:
        trainset = datasets.CIFAR10(root='/data/' + student_number + '/dataset', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=workers)

        testset = datasets.CIFAR10(root='/data/' + student_number + '/dataset', train=False,
                                   download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=workers)
    else:
        trainset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=workers)

        testset = datasets.CIFAR10(root='./data', train=False,
                                   download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ###########################################################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    ###########################################################################################
    if include_visuals:
        import matplotlib.pyplot as plt
        import numpy as np

        # functions to show an image

        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    if include_visuals:
        # show images
        imshow(torchvision.utils.make_grid(images))
    print(len(trainloader))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ###########################################################################################

    print(model_names)
    model = models.__dict__[network_architecture](pretrained=pretrain)
    print("Model %s Loaded" % (network_architecture))

    ###########################################################################################
    #   Model conditional modifications
    #
    # Squeezenet

    if network_architecture == 'squeezenet1_0' or network_architecture == 'squeezenet1_1':
        model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
        model.classifier[3] = nn.AvgPool2d(kernel_size=4, stride=4)
        # model.classifier = torch.nn.Sequential(*(list(model.classifier.children())), nn.LogSoftmax(dim=1))
        # model.classifier = torch.nn.Sequential(*(list(model.classifier.children())), nn.Linear(in_features=1024, out_features=10, bias=True))

    # Resnet

    if network_architecture == 'resnet18' or network_architecture == 'resnet34':
        model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    if network_architecture == 'resnet50' or network_architecture == 'wide_resnet50_2'\
            or network_architecture == 'wide_resnet101_2':
        model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

    # VGG

    if network_architecture == 'vgg11' or network_architecture == 'vgg11_bn' \
            or network_architecture == 'vgg13' or network_architecture == 'alexnet' \
            or network_architecture == 'vgg13_bn' or network_architecture == 'vgg16' \
            or network_architecture == 'vgg16_bn' or network_architecture == 'vgg19'\
            or network_architecture == 'vgg19_bn':
        model.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)

    # Densenet & GoogleNet

    if network_architecture == 'densenet121' or network_architecture == 'googlenet':
        model.classifier = nn.Linear(in_features=1024, out_features=10, bias=True)
    if network_architecture == 'densenet161':
        model.classifier = nn.Linear(in_features=2208, out_features=10, bias=True)
    if network_architecture == 'densenet169':
        model.classifier = nn.Linear(in_features=1664, out_features=10, bias=True)
    if network_architecture == 'densenet201':
        model.classifier = nn.Linear(in_features=1920, out_features=10, bias=True)

    # Mobile Net

    if network_architecture == 'mobilenet_v2' or network_architecture == 'mnasnet0_5'\
            or network_architecture == 'mnasnet0_75' or network_architecture == 'mnasnet1_0'\
            or network_architecture == 'mnasnet1_3':
        model.classifier = nn.Linear(in_features=1280, out_features=10, bias=True)

    print(model)
    print("Model %s Reshaped" % network_architecture)

    ###########################################################################################
    # Send to GPU if available
    if use_cuda:
        print("Sending model to GPU")
        model.to(device)

    ###########################################################################################
    if freeze_layers:
        print("Freezing Layers")
        for child in model.features.children():
            for p in child.parameters():
                p.requires_grad = False

        # Max pooling layer
        for child in model.features[11].children():
            for p in child.parameters():
                p.requires_grad = True

        # Fire: Conv2D layer
        for child in model.features[12].children():
            for p in child.parameters():
                p.requires_grad = True
    ###########################################################################################
    # Dropout

    if dropout:
        model = nn.Dropout(0.5)  # TODO: This is completely wrong

    ###########################################################################################

    criterion = nn.CrossEntropyLoss()
    if optimizer_choice == 1:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_choice == 2:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_choice == 3:
        optimizer = optim.ASGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_choice == 4:
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("Learning Rate: %s, Weight Decay: %s, Momentum: %s" % (learning_rate, weight_decay, momentum))
    print("Defined %s Optimizer" % type(optimizer))

    ###########################################################################################

    start_time = time.time()
    print('Starting Training at %s' % start_time)
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i > trainset_size:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_loss += running_loss
            if i % mini_batch_print == mini_batch_print-1:  # print every 500 mini-batch_scripts

                print('[%d, %5d] loss: %f' %
                      (epoch + 1, i + 1, running_loss / mini_batch_print)) # Printing %.3f and dividing by const 200 not mini_batch_size
                print(time.time() - start_time)
                # print(outputs.max(1))
                running_loss = 0.0
        print('Epoch [%d] loss: %f' % (epoch + 1, epoch_loss))

    print('Finished Training')

    ###########################################################################################
    if peregrine:

        PATH = '/data/' + student_number + '/trained-models/'
        Path(PATH).mkdir(parents=True, exist_ok=True)
        PATH = PATH + network_architecture + str(datetime.now())

    else:
        PATH = './cifar_squeezenet_SCtest.pth'
    torch.save(model.state_dict(), PATH)
    print("Saving model to %s" % PATH)

    ###########################################################################################

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    if include_visuals:
        # print images
        imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ###########################################################################################
    if load_from_memory:
        # TODO: Adjust this to be dynamic to model architecture
        model = models.squeezenet1_0()
        model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
        model.load_state_dict(torch.load(PATH))

    ###########################################################################################
    # Send testing images and loaded model to GPU

    if use_cuda:
        print("Sending data to GPU")
        images, labels = images.to(device), labels.to(device)
        print("Sending model to GPU")
        model.to(device)

    ###########################################################################################

    model.eval()
    outputs = model(images)
    print(outputs)

    ###########################################################################################

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    ###########################################################################################

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            if i > testset_size:
                break
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the ' + str(testset_size) + ' test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    print("###########################################################################################################")
    model_archi = -1
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    desc = ""
    for index, mn in enumerate(model_names):
        desc += ("| %s = %s |" % (index, mn))

    parser = argparse.ArgumentParser(description='Choose an architecture:  ' + desc)
    parser.add_argument('--cuda', '-c', dest='use_cuda', action='store_true',
                        default=False,
                        help='Enable CUDA GPU processing (Default: False)')
    parser.add_argument('--peregrine', '-p', dest='peregrine', action='store_true',
                        default=False,
                        help='Set condition for Peregrine Environment (Default: False)')
    parser.add_argument('--visual', '-v', dest='include_visuals', action='store_true',
                        default=False,
                        help='Enable matplotlib visuals (Default: False)')
    parser.add_argument('--normalise', '-n', dest='normalise', action='store_true',
                        default=False,
                        help='Normalise data before training (Default: False)')
    parser.add_argument('--load', '-l', dest='load_from_memory', action='store_true',
                        default=False,
                        help='Load the model from memory (Default: False)')
    parser.add_argument('--pretrain', '-pt', dest='pretrain', action='store_true',
                        default=False,
                        help='Load the model from memory (Default: False)')
    parser.add_argument('--batch_size', '-b', dest='batch_size', type=int,
                        default=4,
                        help='Batch Size (Default: 4)')
    parser.add_argument('--workers', '-w', dest='workers', type=int,
                        default=2,
                        help='Number of Workers (Default: 2)')
    parser.add_argument('--model', '-m', dest='model_archi', type=int,
                        default=23,
                        help='Model Architecture (Default: 14 = resnet18)')
    parser.add_argument('--train_size', '-ts', dest='trainset_size', type=int,
                        default=20000,
                        help='Set train set size; len(testset) = ts / 5 (Default: 20000)')
    parser.add_argument('--epochs', '-ep', dest='epochs', type=int,
                        default=2,
                        help='Number of training epochs (Default: 2)')
    parser.add_argument('--learning_rate', '-lr', dest='learning_rate', type=float,
                        default=0.001,
                        help='Learning Rate (Default: 0.001)')
    parser.add_argument('--momentum', '-mo', dest='momentum', type=float,
                        default=0,
                        help='Momentum (Default: 0)')
    parser.add_argument('--weight_decay', '-wd', dest='weight_decay', type=float,
                        default=0,
                        help='Weight Decay (Default: 0)')
    parser.add_argument('--optimizer_choice', '-op', dest='optimizer_choice', type=int,
                        default=1,
                        help='Optimizer Choice (Default: 1)')
    args = parser.parse_args()

    print(args.__dict__)

    # freeze_layers = args.__dict__['freeze_layers']
    # dropout = args.__dict__['dropout']
    model_archi = args.__dict__['model_archi']
    batch_size = args.__dict__['batch_size']
    workers = args.__dict__['workers']
    normalise = args.__dict__['normalise']
    include_visuals = args.__dict__['include_visuals']
    use_cuda = args.__dict__['use_cuda']
    peregrine = args.__dict__['peregrine']
    trainset_size = args.__dict__['trainset_size']
    load_from_memory = args.__dict__['load_from_memory']
    pretrain = args.__dict__['pretrain']
    learning_rate = args.__dict__['learning_rate']
    epochs = args.__dict__['epochs']
    momentum = args.__dict__['momentum']
    weight_decay = args.__dict__['weight_decay']
    optimizer_choice = args.__dict__['optimizer_choice']

    print(args)
    test(model_names[model_archi])
