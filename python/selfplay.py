from datetime import datetime
import minishogilib
import numpy as np
import time

import gamerecord


class SelfplayConfig:
    def __init__(self):
        self.max_moves = 512

        # playout cap oscillation
        self.playout_cap_oscillation = False
        self.N = 800
        self.n = 128
        self.oscillation_frac = 0.25

        self.stop_with_checkmate = True

def run(nn, search, config, verbose=False):
    position = minishogilib.Position()
    position.set_start_position()

    game_record = gamerecord.GameRecord()

    for _ in range(config.max_moves):
        is_repetition, is_check_repetition = position.is_repetition()
        if is_check_repetition:
            game_record.winner = 1 - position.get_side_to_move()
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
            best_move = checkmate_move
            if not config.stop_with_checkmate:
                search.mcts.clear()

        else:
            if config.playout_cap_oscillation:
                if np.random.rand() < config.oscillation_frac:
                    search.config.simulation_num = config.N
                    search.config.forced_playouts = True
                    search.config.use_dirichlet = True
                    search.config.reuse_tree = False
                    search.config.target_pruning = True
                    search.config.immediate = False

                else:
                    search.config.simulation_num = config.n
                    search.config.forced_playouts = False
                    search.config.use_dirichlet = False
                    search.config.reuse_tree = True
                    search.config.target_pruning = False
                    search.config.immediate = True

            root = search.run(position, nn)
            best_move = search.best_move(root)

        if verbose:
            if checkmate:
                print('checkmate!')
            else:
                search.print(root)

        elapsed = time.time() - start_time

        position.do_move(best_move)

        game_record.sfen_kif.append(best_move.sfen())
        if checkmate:
            game_record.mcts_result.append(
                (1, 1.0, [(checkmate_move.sfen(), 1)]))

            game_record.learning_target_plys.append(game_record.ply)

        else:
            game_record.mcts_result.append(search.dump(root))

            if not config.playout_cap_oscillation or search.config.simulation_num >= config.N:
                game_record.learning_target_plys.append(game_record.ply)

        game_record.ply += 1

        if verbose:
            print('--------------------')
            position.print()
            print(best_move)
            print('time:', elapsed)
            print('--------------------')

        if checkmate and config.stop_with_checkmate:
            break

    game_record.timestamp = int(datetime.now().timestamp())
    return game_record
