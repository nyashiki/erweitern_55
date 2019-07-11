use pyo3::prelude::*;

use types::*;

#[pyclass]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Move {
    pub piece: Piece,
    pub from: usize,          // 移動元 (持ち駒を打つ場合には、打つ場所)
    pub direction: Direction, // 移動方向
    pub amount: usize,        // 移動量 (0の場合には持ち駒)
    pub to: usize,            // 移動先
    pub promotion: bool,      // 成/不成
    pub capture_piece: Piece, // 取った相手の駒
}

#[pymethods]
impl Move {
    pub fn sfen(&self) -> String {
        const HAND_PIECE_TO_CHAR: [char; 7] = ['E', 'E', 'G', 'S', 'B', 'R', 'P'];

        if self.amount == 0 {
            format!(
                "{}*{}",
                HAND_PIECE_TO_CHAR[self.piece.get_piece_type() as usize],
                square_to_sfen(self.to)
            )
        } else {
            if self.promotion {
                format!("{}{}+", square_to_sfen(self.from), square_to_sfen(self.to))
            } else {
                format!("{}{}", square_to_sfen(self.from), square_to_sfen(self.to))
            }
        }
    }
}

#[pyproto]
impl pyo3::class::basic::PyObjectProtocol for Move {
    fn __repr__(&self) -> PyResult<String> {
        Ok(self.sfen())
    }
}

impl Move {
    pub fn board_move(
        piece: Piece,
        from: usize,
        direction: Direction,
        amount: usize,
        to: usize,
        promotion: bool,
        capture_piece: Piece,
    ) -> Move {
        Move {
            piece: piece,
            from: from,
            direction: direction,
            amount: amount,
            to: to,
            promotion: promotion,
            capture_piece: capture_piece,
        }
    }

    pub fn hand_move(piece: Piece, to: usize) -> Move {
        Move {
            piece: piece,
            from: 0,
            direction: Direction::N, // 不使用なので仮の値を入れておく
            amount: 0,
            to: to,
            promotion: false,
            capture_piece: Piece::NoPiece,
        }
    }
}

pub static NULL_MOVE: Move = Move {
    piece: Piece::NoPiece,
    from: SQUARE_NB,
    direction: Direction::N,
    amount: 0,
    to: 0,
    promotion: false,
    capture_piece: Piece::NoPiece,
};

fn square_to_sfen(square: usize) -> String {
    format!(
        "{}{}",
        "54321".as_bytes()[square % 5 as usize] as char,
        "abcde".as_bytes()[square / 5 as usize] as char
    )
}
