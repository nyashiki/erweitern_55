from datetime import datetime
import minishogilib
import time

import gamerecord

class SelfplayConfig:
    def __init__(self):
        self.num_sampling_moves = 30
        self.max_moves = 512
        self.use_dirichlet = False

def run(nn, search, config, verbose=False):
    position = minishogilib.Position()
    position.set_start_position()

    game_record = gamerecord.GameRecord()

    for _ in range(config.max_moves):
        start_time = time.time()

        checkmate, checkmate_move = position.solve_checkmate_dfs(7)
        if checkmate:
            best_move = checkmate_move
        else:
            root = search.run(position, nn, config.use_dirichlet)
            best_move = search.best_move(root)

        if verbose:
            if checkmate:
                print('checkmate!')
            else:
                search.print(root)

        elapsed = time.time() - start_time

        if best_move.is_null_move():
            game_record.winner = 1 - position.get_side_to_move()

            break

        position.do_move(best_move)

        game_record.ply += 1
        game_record.sfen_kif.append(best_move.sfen())
        if checkmate:
            game_record.mcts_result.append((1, 1.0, [(checkmate_move.sfen(), 1)]))
        else:
            game_record.mcts_result.append(search.dump(root))

        if verbose:
            print('--------------------')
            position.print()
            print(best_move)
            print('time:', elapsed)
            print('--------------------')

    game_record.timestamp = int(datetime.now().timestamp())
    return game_record
