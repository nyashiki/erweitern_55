//! NeuralNetworkに関係のある部分の実装
//!
//! ここでは、NeuralNetworkのForwardやBackpropagationなどを実装するのではなく、
//! tensorflow等の使用を容易にすることを目指す

#[cfg(test)]
use rand::seq::SliceRandom;

use position::Position;
use r#move::*;
use types::*;

use numpy::PyArray1;
use pyo3::prelude::*;

/// NeuralNetworkの入力層に与える形式に変換した際の、チャネル数
///
/// --------------------------------------------------------------
/// Feature                                             # Channels
/// --------------------------------------------------------------
/// P1 piece                                                    10
/// P2 piece                                                    10
/// Repetitions                                                  3
/// P1 prisoner count                                            5
/// P2 prisoner count                                            5
/// --------------------------------------------------------------
/// Color                                                        1
/// Total move count                                             1
/// --------------------------------------------------------------
/// Total                  (10 + 10 + 3 + 5 + 5) * HISTORY_NUM + 2
/// --------------------------------------------------------------

#[pymethods]
impl Position {
    /// \[チャネル\]\[y座標\]\[x座標\]の形式で返す
    pub fn to_nninput(&self, py: Python) -> Py<PyArray1<f32>> {
        const HISTORY: usize = 4;
        const CHANNEL_NUM_PER_HISTORY: usize = 10 + 10 + 3 + 5 + 5;
        const CHANNEL_NUM: usize = CHANNEL_NUM_PER_HISTORY * HISTORY + 2;

        let mut input_layer = [0f32; CHANNEL_NUM * SQUARE_NB];

        let mut position = *self;

        for h in 0..HISTORY {
            if h > 0 {
                // 局面を1手戻す
                position.undo_move();
            }

            for i in 0..SQUARE_NB {
                // 盤上の駒を設定
                if position.board[i] != Piece::NoPiece {
                    if self.side_to_move == Color::White {
                        input_layer[(2
                            + h * CHANNEL_NUM_PER_HISTORY
                            + piece_to_sequential_index(position.board[i]))
                            * SQUARE_NB
                            + i] = 1f32;
                    } else {
                        // 後手番の場合には、盤面を回転させて設定する
                        input_layer[(2
                            + h * CHANNEL_NUM_PER_HISTORY
                            + piece_to_sequential_index(position.board[i].get_op_piece()))
                            * SQUARE_NB
                            + (SQUARE_NB - i - 1)] = 1f32;
                    }
                }

                // 繰り返し回数を設定
                input_layer[(2 + h * CHANNEL_NUM_PER_HISTORY + 20 + position.get_repetition())
                    * SQUARE_NB
                    + i] = 1f32;

                // 持ち駒を設定
                for piece_type in HAND_PIECE_TYPE_ALL.iter() {
                    input_layer[(2 + h * CHANNEL_NUM_PER_HISTORY + 23 + *piece_type as usize
                        - 2)
                        * SQUARE_NB
                        + i] =
                        position.hand[self.side_to_move as usize][*piece_type as usize - 2] as f32;
                    input_layer[(2 + h * CHANNEL_NUM_PER_HISTORY + 28 + *piece_type as usize
                        - 2)
                        * SQUARE_NB
                        + i] = position.hand[self.side_to_move.get_op_color() as usize]
                        [*piece_type as usize - 2] as f32;
                }
            }

            if position.ply == 0 {
                break;
            }
        }

        for i in 0..SQUARE_NB {
            // 手番を設定
            if self.side_to_move == Color::Black {
                input_layer[i] = 1f32;
            }

            // 手数を設定
            input_layer[SQUARE_NB + i] = self.ply as f32;
        }

        return PyArray1::from_slice(py, &input_layer).to_owned();
    }
}

#[pymethods]
impl Move {
    pub fn to_policy_index(&self) -> usize {
        let c: Color = self.piece.get_color();

        let index = if self.amount == 0 {
            if c == Color::White {
                (64 + self.get_hand_index(), self.to)
            } else {
                (64 + self.get_hand_index(), SQUARE_NB - 1 - self.to)
            }
        } else {
            if self.get_promotion() {
                if c == Color::White {
                    (32 + 4 * self.direction as usize + self.amount - 1, self.from)
                } else {
                    (
                        32 + 4 * ((self.direction as usize + 4) % 8) + self.amount - 1,
                        SQUARE_NB - 1 - self.from,
                    )
                }
            } else {
                if c == Color::White {
                    (4 * self.direction as usize + self.amount - 1, self.from)
                } else {
                    (
                        4 * ((self.direction as usize + 4) % 8) + self.amount - 1,
                        SQUARE_NB - 1 - self.from,
                    )
                }
            }
        };

        return index.0 * 25 + index.1;
    }
}

#[cfg(test)]
fn index_to_move(position: &Position, index: usize) -> Move {
    let mut moves: std::vec::Vec<Move> = Vec::new();

    if index >= 64 * 25 {
        for i in 0..5 {
            for j in 0..SQUARE_NB {
                let temp = if position.side_to_move == Color::White {
                    (64 + i) * 25 + j
                } else {
                    (64 + i) * 25 + (SQUARE_NB - j - 1)
                };

                if temp == index {
                    moves.push(Move::hand_move(
                        HAND_PIECE_TYPE_ALL[i].get_piece(position.side_to_move),
                        j,
                    ));
                }
            }
        }
    } else {
        for direction in 0..8 {
            for amount in 0..4 {
                for i in 0..SQUARE_NB {
                    for promotion in 0..2 {
                        let temp = if position.side_to_move == Color::White {
                            (32 * promotion + ((direction * 4) + amount)) * 25 + i
                        } else {
                            (32 * promotion + ((((direction + 4) % 8) * 4) + amount)) * 25
                                + (SQUARE_NB - i - 1)
                        };

                        if temp == index {
                            moves.push(Move::board_move(
                                Piece::NoPiece,
                                i,
                                DIRECTION_ALL[direction],
                                amount + 1,
                                0,
                                promotion != 0,
                                Piece::NoPiece,
                            ));
                        }
                    }
                }
            }
        }
    }

    assert_eq!(moves.len(), 1);
    return moves[0];
}

#[test]
fn to_policy_index_test() {
    ::bitboard::init();

    const LOOP_NUM: i32 = 10000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            let moves = position.generate_moves();

            for m in &moves {
                let index = m.to_policy_index();
                let move_from_index = index_to_move(&position, index);

                assert_eq!(m.amount, move_from_index.amount);
                assert_eq!(m.direction, move_from_index.direction);

                if m.amount == 0 {
                    assert_eq!(m.to, move_from_index.to);
                } else {
                    assert_eq!(m.from, move_from_index.from);
                    assert_eq!(m.promotion, move_from_index.promotion);
                }
            }

            // ランダムに局面を進める
            if moves.len() == 0 {
                break;
            }

            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

fn piece_to_sequential_index(piece: Piece) -> usize {
    match piece {
        Piece::WKing => 0,
        Piece::WGold => 1,
        Piece::WSilver => 2,
        Piece::WBishop => 3,
        Piece::WRook => 4,
        Piece::WPawn => 5,
        Piece::WSilverX => 6,
        Piece::WBishopX => 7,
        Piece::WRookX => 8,
        Piece::WPawnX => 9,

        Piece::BKing => 10,
        Piece::BGold => 11,
        Piece::BSilver => 12,
        Piece::BBishop => 13,
        Piece::BRook => 14,
        Piece::BPawn => 15,
        Piece::BSilverX => 16,
        Piece::BBishopX => 17,
        Piece::BRookX => 18,
        Piece::BPawnX => 19,

        Piece::NoPiece => 20,
    }
}
