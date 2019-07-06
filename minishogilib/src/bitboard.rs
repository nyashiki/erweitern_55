use types::*;
use position::*;

pub type Bitboard = u32;

pub static mut ADJACENT_ATTACK: [[Bitboard; Piece::BPawnX as usize + 1]; SQUARE_NB] = [[0; Piece::BPawnX as usize + 1]; SQUARE_NB];

pub fn init() {
    let mut position: Position = Position::empty_board();

    for i in 0..SQUARE_NB {
        for piece in PIECE_ALL.iter() {
            position.board[i] = *piece;
            position.side_to_move = piece.get_color();

            let moves = position.generate_moves(true, false);

            for m in moves {
                if m.amount != 1 {
                    continue;
                }

                unsafe {
                    ADJACENT_ATTACK[i][*piece as usize] |= 1 << m.to;
                }
            }
        }
        position.board[i] = Piece::NoPiece;
    }
}
