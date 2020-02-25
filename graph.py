from matplotlib import pyplot as plt
import numpy as np
from os.path import dirname, basename, isfile, join
import glob

files = []


def open_files():
    global files
    modules = glob.glob(join(dirname(__file__), "experiment1/*.npy"))
    __all__ = [basename(f)[:] for f in modules if isfile(f) and not f.endswith('*.out')]
    files = __all__


def run():
    results = []
    for f in files:
        string = 'experiment1/' + f

        results.append(np.load(string))
        # with np.load(string) as data:
        #     results.append(data)

    print(files)
    print(int(results[0][2][1]))

    for index, run in enumerate(results[0:]):
        x = []
        y = []
        fig, axs = plt.subplots(1)
        axs.set_ylabel('Loss')
        axs.set_xlabel('Training Steps')
        for r in run:
            if r[1] != -1:
                x.append((int(r[0] * 12500)) + r[1])
                y.append(r[2])
            # print(r[1:])
        axs.plot(x, y, label=files[index])
        axs.legend()
        plt.show()


if __name__ == '__main__':
    open_files()
    run()