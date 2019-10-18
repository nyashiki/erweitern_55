import cairosvg
import io
import matplotlib.pyplot as plt
import minishogilib
import numpy as np
import PIL
import simplejson
import sys


def main():
    record_path = sys.argv[1]

    target_positions = [
        ['4e4d'],
        ['2e3d'],
        ['3e2d'],
        ['3e4d'],
        ['3e3d'],
        ['2e1d']
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

    elements = []
    elements.append('<html>')
    elements.append('<head>')
    elements.append('<meta charset="utf-8">')
    elements.append('<title>Analyzer</title>')
    elements.append('</head>')
    elements.append('<body>')

    elements.append('<table>')
    for (i, pf) in enumerate(position_frequency):
        plt.clf()

        position = minishogilib.Position()
        position.set_start_position()

        for m in target_positions[i]:
            move = position.sfen_to_move(m)
            position.do_move(move)

        elements.append('<tr>')

        elements.append('<td>')
        elements.append(position.to_svg())
        elements.append('</td>')

        plt.ylim([0, 100])
        plt.grid(linestyle='--')
        y = np.array(pf) * 100
        plt.plot(range(len(pf)), y)

        f = io.BytesIO()
        plt.savefig(f, format='svg')
        elements.append('<td>')
        elements.append(f.getvalue().decode('utf-8'))
        elements.append('</td>')

        elements.append('</tr>')

    elements.append('</table>')

    elements.append('</body>')
    elements.append('</html>')

    with open('./index.html', 'w') as html:
        html.write('\n'.join(elements))


if __name__ == '__main__':
    main()
