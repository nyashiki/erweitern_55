import matplotlib.pyplot as plt
import simplejson
import sys


def main():
    record_path = sys.argv[1]

    target_positions = [
        ['3e2d', '3a4b', '2e3d'],
        ['3e4d', '3a4b', '2e3d'],
        ['4e4d', '2a2b', '2e3d'],
        ['4e4d', '4a3b', '2e3d'],
    ]

    position_counts = [0 for _ in target_positions]
    position_frequency = [[] for _ in target_positions]

    with open(record_path) as f:
        counter = 0

        while True:
            line = f.readline()

            if not line:
                break

            data = simplejson.loads(line)

            for (i, target_position) in enumerate(target_positions):
                if data['sfen_kif'][:len(target_position)] == target_position:
                    position_counts[i] += 1

            counter += 1

            if counter == 100:
                for i in range(len(target_positions)):
                    position_frequency[i].append(position_counts[i] / counter)

                position_counts = [0 for _ in target_positions]
                counter = 0

    for (i, pf) in enumerate(position_frequency):
        plt.subplot(len(target_positions), 1, (i + 1))
        plt.plot(range(len(pf)), pf)

    plt.show()

if __name__ == '__main__':
    main()
