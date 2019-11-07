import datetime
import minishogilib
import numpy as np
from scipy.special import softmax
import sys
import tensorflow as tf

from erweitern_55 import network


def main():
    record_file = './erweitern_55/records.json'
    weights_file = './erweitern_55/weights.h5'
    RECENT_GAMES = 500000

    # Construct two neural networks.
    teacher_net = network.Network('gpu', 'DenseNet')
    student_net = network.Network('gpu', 'CNN')

    # Teacher network loads weights.
    teacher_net.load(weights_file)

    # Training dataset.
    sys.stdout.write('Loading teacher data')
    sys.stdout.flush()
    reservoir = minishogilib.Reservoir(record_file, RECENT_GAMES)
    reservoir.load(record_file)
    sys.stdout.write(' ... ok.\n')
    sys.stdout.flush()

    BATCH_SIZE = 4096

    position = minishogilib.Position()
    position.set_start_position()
    init_input = student_net.get_input(position, True)

    log_file = open('distillation.txt', 'w')

    # Change the optimizer of student network to Adam.
    student_net.model.optimizer = tf.keras.optimizers.Adam()

    while True:
        # Sample training data.
        sample = reservoir.sample(BATCH_SIZE)

        # Reshape it into nn format.
        ins = np.reshape(sample[0], [BATCH_SIZE] + teacher_net.input_shape)

        # Get teacher's output.
        teacher_policy, teacher_value = teacher_net.predict(ins)
        teacher_policy = softmax(teacher_policy, axis=1)

        # Train the student network.
        loss = student_net.step(ins, teacher_policy, teacher_value)
        init_policy, init_value = student_net.predict(init_input)

        if student_net.iter() % 5000 == 0:
            student_net.save('./distillation/iter_{}.h5'.format(student_net.iter()))

        log_file.write('{}, {}, {}, {}, {}, {}\n'.format(datetime.datetime.now(
            datetime.timezone.utc), student_net.iter(), loss['loss'], loss['policy_loss'], loss['value_loss'], init_value[0][0]))
        log_file.flush()

if __name__ == '__main__':
    main()
