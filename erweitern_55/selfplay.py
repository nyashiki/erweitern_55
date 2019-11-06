from datetime import datetime
import minishogilib
import numpy as np
import time

import gamerecord


def run(nn, search, verbose=False, num_sampling_moves=30, max_moves=512, playout_cap_oscillation={'enable': False, 'N': 800, 'n': 128, 'frac': 0.25}, stop_with_checkmate=False):
    position = minishogilib.Position()
    position.set_start_position()

    game_record = gamerecord.GameRecord()

    for _ in range(max_moves):
        is_repetition, is_check_repetition = position.is_repetition()
        if is_check_repetition:
            game_record.winner = position.get_side_to_move()
            break
        elif is_repetition:
            game_record.winner = 1
            break

        moves = position.generate_moves()
        if len(moves) == 0:
            game_record.winner = 1 - position.get_side_to_move()
            break

        start_time = time.time()

        checkmate, checkmate_move = position.solve_checkmate_dfs(7)

        if checkmate:
            next_move = checkmate_move
            if not stop_with_checkmate:
                search.mcts.clear()

        else:
            if playout_cap_oscillation['enable']:
                if np.random.rand() < playout_cap_oscillation['frac']:
                    search.config.simulation_num = playout_cap_oscillation['N']
                    search.config.forced_playouts = True
                    search.config.use_dirichlet = True
                    search.config.reuse_tree = False
                    search.config.target_pruning = True
                    search.config.immediate = False

                else:
                    search.config.simulation_num = playout_cap_oscillation['n']
                    search.config.forced_playouts = False
                    search.config.use_dirichlet = False
                    search.config.reuse_tree = True
                    search.config.target_pruning = False
                    search.config.immediate = True

            root = search.run(position, nn)

            if position.get_ply() < num_sampling_moves:
                next_move = search.softmax_sample(root, 1.0)
            else:
                next_move = search.best_move(root)

        if verbose:
            if checkmate:
                print('checkmate!')
            else:
                search.print(root)

        elapsed = time.time() - start_time

        position.do_move(next_move)

        game_record.sfen_kif.append(next_move.sfen())
        if checkmate:
            game_record.mcts_result.append(
                (1, 1.0, [(checkmate_move.sfen(), 1)]))

            game_record.learning_target_plys.append(game_record.ply)

        else:
            game_record.mcts_result.append(search.dump(root))

            if not playout_cap_oscillation['enable'] or search.config.simulation_num >= playout_cap_oscillation['N']:
                game_record.learning_target_plys.append(game_record.ply)

        game_record.ply += 1

        if verbose:
            print('--------------------')
            position.print()
            print(next_move)
            print('time:', elapsed)
            print('--------------------')

        if checkmate and stop_with_checkmate:
            game_record.winner = 1 - position.get_side_to_move()
            break

    game_record.timestamp = int(datetime.now().timestamp())
    return game_record
