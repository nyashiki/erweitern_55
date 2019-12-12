import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def main(loss_file, output_file):
    with open(loss_file) as f:
        iteration, loss_sum, loss_policy, loss_value, init_value = [], [], [], [], []

        line = f.readline()
        while line:
            split = line.split(',')
            i, ls, lp, lv, iv = int(split[1]), float(split[2]), float(
                split[3]), float(split[4]), float(split[5])

            iteration.append(i)
            loss_sum.append(ls)
            loss_policy.append(lp)
            loss_value.append(lv)
            init_value.append(iv)

            line = f.readline()

        interval = 10

        fig, ax = plt.subplots(2, 1, figsize=(16, 10))

        # loss
        ax[0].plot(iteration[::interval], loss_sum[::interval], linestyle='-',
                   label='total loss', alpha=0.7)
        ax[0].plot(iteration[::interval], loss_policy[::interval],
                   linestyle='--', label='policy loss', alpha=0.7)
        ax[0].plot(iteration[::interval], loss_value[::interval],
                   linestyle=':', label='value loss', alpha=0.7)
        ax[0].set_xlabel('iterations')
        ax[0].set_title('losses')
        ax[0].grid(linestyle='-')
        ax[0].legend(frameon=True)

        # init value
        ax[1].plot(iteration[::interval], init_value[::interval], alpha=0.7)
        ax[1].set_xlabel('iterations')
        ax[1].set_title('value output at the initial position')
        ax[1].grid(linestyle='-')
        ax[1].set_ylim([-1, 1])

        fig.tight_layout()

        fig.savefig(output_file)


if __name__ == '__main__':
    plt.rcParams['font.size'] = 14
    plt.style.use('seaborn')

    main(sys.argv[1], sys.argv[2])
