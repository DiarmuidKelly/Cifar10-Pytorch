from matplotlib import pyplot as plt
import numpy as np
from os.path import dirname, basename, isfile, join
import glob

files = []


def open_files():
    global files
    modules = glob.glob(join(dirname(__file__), "cleaned/*.npy"))
    __all__ = [basename(f)[:] for f in modules if isfile(f) and not f.endswith('*.out')]
    files = __all__


def run():
    results = []
    labels = ['Resnet101 Default', 'Resenet101 Pretrained', 'Resnet50 Default', 'Resnet50 Pretrained',
              'Wide_Resnet101 Default', 'Wide_Resnet101 Pretrained', 'Wide_Resnet50 Default', 'Wide_Resnet50 Pretrained'
              ]
    for f in files:
        string = 'cleaned/' + f

        results.append(np.load(string))
        # with np.load(string) as data:
        #     results.append(data)

    print(files)
    print(int(results[0][2][1]))
    t_x = []
    t_y = []

    for index, run in enumerate(results[0:]):
        x = []
        y = []

        for r in run:
            if r[1] != -1:
                x.append((int(r[0] * 12500)) + r[1])
                y.append(r[2])
            # print(r[1:])
        t_x.append(x)
        t_y.append(y)

    for i in range(0, 4, 2):
        fig, axs = plt.subplots(1)
        axs.set_ylabel('Loss')
        axs.set_xlabel('Training Steps')
        axs.plot(t_x[i], t_y[i], label=labels[i])
        axs.plot(t_x[i+1], t_y[i+1], label=labels[i+1], color='red')
        axs.plot(t_x[i+4], t_y[i+4], label=labels[i+4], color='green')
        axs.plot(t_x[i+5], t_y[i+5], label=labels[i+5], color='orange')
        axs.legend()
        # plt.show()
        plt.savefig('./cleaned/' + labels[i] + '-doubleoverlay.png')
        plt.close()


if __name__ == '__main__':
    open_files()
    run()