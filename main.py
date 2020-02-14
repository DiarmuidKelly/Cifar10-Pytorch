import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim

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
trainset_size = 20000
testset_size = trainset_size / 5
model_names = []

student_number = "s4091221"  # Used for peregrine directory


def test(network_architecture):

    ###########################################################################################

    if normalise:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()])

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
    model = models.__dict__[network_architecture](pretrained=True)
    print("Model %s Loaded" % (network_architecture))
    print(model)

    ###########################################################################################
    #   Model conditional modifications

    if network_architecture == 'squeezenet1_0' or network_architecture == 'squeezenet1_1':
        model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
    if network_architecture == 'resnet18' or network_architecture == 'resnet34':
        model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    if network_architecture == 'resnet50':
        model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    if network_architecture == 'vgg11' or network_architecture == 'vgg11_bn':
        model.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)

    ###########################################################################################
    # Send to GPU if available
    if use_cuda:
        print("Sending training data to GPU")
        dataiter = iter(trainloader)
        for images, labels in dataiter:
            images, labels = images.to(device), labels.to(device)
        print("Sending testing data to GPU")
        dataiter = iter(testloader)
        for images, labels in dataiter:
            images, labels = images.to(device), labels.to(device)
        print("Sending model to GPU")
        model.to(device)

    # print(model)

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

        # for child in model.features.children():
        #     for p in child.parameters():
        #       print(p.requires_grad)

    ###########################################################################################
    # Dropout

    if dropout:
        model = nn.Dropout(0.5) # TODO: This is completely wrong

    ###########################################################################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    print("Defined Optimizer")

    ###########################################################################################
    start_time = time.time()
    print('Starting Training at %s' % (start_time))
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i > trainset_size:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                print(time.time() - start_time)
                running_loss = 0.0

    print('Finished Training')

    ###########################################################################################

    PATH = './cifar_squeezenet_SCtest.pth'

    torch.save(model.state_dict(), PATH)

    ###########################################################################################
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    if include_visuals:
        # print images
        imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ###########################################################################################

    model = models.squeezenet1_0()
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(PATH))

    ###########################################################################################

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
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the ' + str(testset_size) + ' test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
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

    print(args)
    test(model_names[model_archi])
