//! NeuralNetworkに関係のある部分の実装
//!
//! ここでは、NeuralNetworkのForwardやBackpropagationなどを実装するのではなく、
//! tensorflow等の使用を容易にすることを目指す
use position::Position;
use types::*;

use pyo3::prelude::*;

/// 過去何手分の局面を入力層に含めるか
const HISTORY_NUM: usize = 8;

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
const CHANNEL_NUM_PER_HISTORY: usize = 10 + 10 + 3 + 5 + 5;
const CHANNEL_NUM: usize = CHANNEL_NUM_PER_HISTORY * HISTORY_NUM + 2;

#[pymethods]
impl Position {
    /// \[y座標\]\[x座標\]\[チャネル\]の形式で返す
    pub fn to_nninput(&self) -> std::vec::Vec<std::vec::Vec<std::vec::Vec<f32>>> {
        let mut input_layer = vec![vec![vec![0f32; CHANNEL_NUM]; 5]; 5];

        let mut position = *self;

        for history in 0..HISTORY_NUM {
            for i in 0..5 {
                for j in 0..5 {
                    // 盤上の駒を設定
                    if position.board[i * 5 + j] != Piece::NoPiece {
                        if self.side_to_move == Color::White {
                            input_layer[i][j][history * CHANNEL_NUM_PER_HISTORY
                                + piece_to_sequential_index(position.board[i * 5 + j])] = 1f32;
                        } else {
                            // 後手番の場合には、盤面を回転させて設定する
                            input_layer[4 - i][4 - j][history * CHANNEL_NUM_PER_HISTORY
                                + piece_to_sequential_index(
                                    position.board[i * 5 + j].get_op_piece(),
                                )] = 1f32;
                        }
                    }

                    // 繰り返し回数を設定
                    input_layer[i][j]
                        [history * CHANNEL_NUM_PER_HISTORY + 20 + position.get_repetition()] = 1f32;

                    // 持ち駒を設定
                    for piece_type in HAND_PIECE_TYPE_ALL.iter() {
                        input_layer[i][j]
                            [history * CHANNEL_NUM_PER_HISTORY + 23 + *piece_type as usize - 2] =
                            position.hand[position.side_to_move as usize][*piece_type as usize - 2]
                                as f32;
                        input_layer[i][j]
                            [history * CHANNEL_NUM_PER_HISTORY + 28 + *piece_type as usize - 2] =
                            position.hand[position.side_to_move.get_op_color() as usize]
                                [*piece_type as usize - 2] as f32;
                    }
                }
            }

            if position.ply == 0 {
                break;
            }

            // 局面を1手戻す
            position.undo_move();
        }

        for i in 0..5 {
            for j in 0..5 {
                // 手番を設定
                if self.side_to_move == Color::Black {
                    input_layer[i][j][CHANNEL_NUM - 2] = 1f32;
                }

                // 手数を設定
                input_layer[i][j][CHANNEL_NUM - 1] = self.ply as f32;
            }
        }

        return input_layer;
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
