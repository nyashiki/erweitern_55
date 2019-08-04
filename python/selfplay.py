import minishogilib
import time

def run(nn, search, verbose=False):
    position = minishogilib.Position()
    position.set_start_position()

    while True:
        start_time = time.time()

        checkmate, checkmate_move = position.solve_checkmate_dfs(7)
        if checkmate:
            best_move = checkmate_move
        else:
            root = search.run(position, nn)
            best_move = search.best_move(root)

            if verbose:
                search.dump(root)

        elapsed = time.time() - start_time

        if best_move.is_null_move():
            break

        position.do_move(best_move)

        if verbose:
            print('--------------------')
            position.print()
            print(best_move)
            print('time:', elapsed)
            print('--------------------')
