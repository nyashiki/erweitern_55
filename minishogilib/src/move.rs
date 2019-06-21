use pyo3::prelude::*;

use types::*;

#[pyclass]
#[derive(Copy, Clone)]
pub struct Move {
    piece: Piece,
    target: u8,            // 移動元 (持ち駒を打つ場合には、打つ場所)
    direction: Direction,  // 移動方向
    amount: u8,            // 移動量 (0の場合には持ち駒)
    promotion: bool        // 成/不成
}

impl Move {
    pub fn board_move(piece: Piece, target: u8, direction: Direction, amount: u8, promotion: bool) -> Move {
        Move {
            piece: piece,
            target: target,
            direction: direction,
            amount: amount,
            promotion: promotion
        }
    }

    pub fn hand_move(piece: Piece, target: u8) -> Move {
        Move {
            piece: piece,
            target: target,
            direction: Direction::N,  // 不使用なので仮の値を入れておく
            amount: 0,
            promotion: false
        }
    }
}
