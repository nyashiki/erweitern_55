use pyo3::prelude::*;

use types::*;

#[pyclass]
#[derive(Copy, Clone)]
pub struct Move {
    pub piece: Piece,
    pub target: u8,            // 移動元 (持ち駒を打つ場合には、打つ場所)
    pub direction: Direction,  // 移動方向
    pub amount: u8,            // 移動量 (0の場合には持ち駒)
    pub promotion: bool,       // 成/不成
    pub capture_piece: Piece   // 取った相手の駒
}

impl Move {
    pub fn board_move(piece: Piece, target: u8, direction: Direction, amount: u8, promotion: bool, capture_piece: Piece) -> Move {
        Move {
            piece: piece,
            target: target,
            direction: direction,
            amount: amount,
            promotion: promotion,
            capture_piece: capture_piece
        }
    }

    pub fn hand_move(piece: Piece, target: u8) -> Move {
        Move {
            piece: piece,
            target: target,
            direction: Direction::N,  // 不使用なので仮の値を入れておく
            amount: 0,
            promotion: false,
            capture_piece: Piece::NoPiece
        }
    }
}

pub static NULL_MOVE: Move = Move {
    piece: Piece::NoPiece,
    target: SQUARE_NB as u8,
    direction: Direction::N,
    amount: 0,
    promotion: false,
    capture_piece: Piece::NoPiece
};
