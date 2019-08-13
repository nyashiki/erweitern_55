import numpy as np

def policy_max_move(position, nn):
    moves = position.generate_moves()

    nninput = np.reshape(position.to_nninput(), (1, network.INPUT_CHANNEL, 5, 5))
    policy, value = nn.predict(nninput)

    policy_max = -1
    policy_max_move = None
    for move in moves:
        p = policy[move.to_policy_index()]
        if  p > policy_max:
            policy_max = p
            policy_max_move = move

    return policy_max_move

def value_max_move(position, nn):
    moves = position.generate_moves()

    nninputs = np.zeros((len(moves), network.INPUT_CHANNEL, 5, 5))
    for (i, move) in enumerate(moves):
        pos = position.copy(False)
        pos.do_move(move)
        nninputs[i] = np.reshape(pos.to_nninput(), (network.INPUT_CHANNEL, 5, 5))

    policy, value = nn.predict(nninputs)
    value = (value + 1) / 2

    value_max = -1
    value_max_move = None
    for (i, move) in enumerate(moves):
        v = value[i][0]
        if v > value_max:
            value_max = v
            value_max_move = move

    return value_max_move
