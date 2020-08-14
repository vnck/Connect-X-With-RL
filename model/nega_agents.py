from kaggle_environments.envs.connectx.connectx import *

EMPTY = 0
def return_nega(depth):
    def negamax_agent(obs, config):
        columns = config.columns
        rows = config.rows
        size = rows * columns

        # Due to compute/time constraints the tree depth must be limited.
        max_depth = depth

        def negamax(board, mark, depth):
            moves = sum(1 if cell != EMPTY else 0 for cell in board)

            # Tie Game
            if moves == size:
                return (0, None)

            # Can win next.
            for column in range(columns):
                if board[column] == EMPTY and is_win(board, column, mark, config, False):
                    return ((size + 1 - moves) / 2, column)

            # Recursively check all columns.
            best_score = -size
            best_column = None
            for column in range(columns):
                if board[column] == EMPTY:
                    # Max depth reached. Score based on cell proximity for a clustering effect.
                    if depth <= 0:
                        row = max(
                            [
                                r
                                for r in range(rows)
                                if board[column + (r * columns)] == EMPTY
                            ]
                        )
                        score = (size + 1 - moves) / 2
                        if column > 0 and board[row * columns + column - 1] == mark:
                            score += 1
                        if (
                            column < columns - 1
                            and board[row * columns + column + 1] == mark
                        ):
                            score += 1
                        if row > 0 and board[(row - 1) * columns + column] == mark:
                            score += 1
                        if row < rows - 2 and board[(row + 1) * columns + column] == mark:
                            score += 1
                    else:
                        next_board = board[:]
                        play(next_board, column, mark, config)
                        (score, _) = negamax(next_board,
                                             1 if mark == 2 else 2, depth - 1)
                        score = score * -1
                    if score > best_score or (score == best_score and choice([True, False])):
                        best_score = score
                        best_column = column

            return (best_score, best_column)

        _, column = negamax(obs.board[:], obs.mark, max_depth)
        if column == None:
            column = choice([c for c in range(columns) if obs.board[c] == EMPTY])
        return column
    return negamax_agent