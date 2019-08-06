from datetime import datetime
import minishogilib
import time

import gamerecord


class SelfplayConfig:
    def __init__(self):
        self.max_moves = 512


def run(nn, search, config, verbose=False):
    position = minishogilib.Position()
    position.set_start_position()

    game_record = gamerecord.GameRecord()

    for _ in range(config.max_moves):
        moves = position.generate_moves()
        if len(moves) == 0:
            game_record.winner = 1 - position.get_side_to_move()

            break

        start_time = time.time()

        checkmate, checkmate_move = position.solve_checkmate_dfs(7)
        if checkmate:
            best_move = checkmate_move
        else:
            root = search.run(position, nn)
            best_move = search.best_move(root)

        if verbose:
            if checkmate:
                print('checkmate!')
            else:
                search.print(root)

        elapsed = time.time() - start_time

        position.do_move(best_move)

        game_record.ply += 1
        game_record.sfen_kif.append(best_move.sfen())
        if checkmate:
            game_record.mcts_result.append(
                (1, 1.0, [(checkmate_move.sfen(), 1)]))
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
