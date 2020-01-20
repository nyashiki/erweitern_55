import cairosvg
import io
import matplotlib
import matplotlib.pyplot as plt
import minishogilib
import numpy as np
import PIL
import seaborn
import simplejson
import sys

matplotlib.use('Agg')


def main():
    record_path = sys.argv[1]

    ply_counts = []

    sample_iter = 200
    ply_sum = 0

    with open(record_path) as f:
        counter = 0

        while True:
            line = f.readline()

            if not line:
                break

            data = simplejson.loads(line)

            ply_sum += data['ply']
            counter += 1

            if counter == sample_iter:
                ply_counts.append(ply_sum / sample_iter)
                ply_sum = 0
                counter = 0

    plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='x')

    plt.plot(range(len(ply_counts) * sample_iter)
             [::sample_iter], ply_counts, alpha=0.7)
    plt.xlabel('iterations')
    plt.ylabel('ply')

    plt.savefig(sys.argv[2])


if __name__ == '__main__':
    plt.style.use('seaborn')

    main()
