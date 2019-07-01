use pyo3::prelude::*;

use types::*;

#[pyclass]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Move {
    pub piece: Piece,
    pub from: u8,              // 移動元 (持ち駒を打つ場合には、打つ場所)
    pub direction: Direction,  // 移動方向
    pub amount: u8,            // 移動量 (0の場合には持ち駒)
    pub to: u8,                // 移動先
    pub promotion: bool,       // 成/不成
    pub capture_piece: Piece   // 取った相手の駒
}

impl Move {
    pub fn board_move(piece: Piece, from: u8, direction: Direction, amount: u8, to: u8, promotion: bool, capture_piece: Piece) -> Move {
        Move {
            piece: piece,
            from: from,
            direction: direction,
            amount: amount,
            to: to,
            promotion: promotion,
            capture_piece: capture_piece
        }
    }

    pub fn hand_move(piece: Piece, to: u8) -> Move {
        Move {
            piece: piece,
            from: 0,
            direction: Direction::N,  // 不使用なので仮の値を入れておく
            amount: 0,
            to: to,
            promotion: false,
            capture_piece: Piece::NoPiece
        }
    }
}

pub static NULL_MOVE: Move = Move {
    piece: Piece::NoPiece,
    from: SQUARE_NB as u8,
    direction: Direction::N,
    amount: 0,
    to: 0,
    promotion: false,
    capture_piece: Piece::NoPiece
};
