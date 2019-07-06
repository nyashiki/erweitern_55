use bitintr::Pext;

use types::*;
use position::*;

pub type Bitboard = u32;

lazy_static! {
    /// 近接の利きを保持するbitboard
    /// ADJACENT_ATTACK[piece][square]として参照する
    static ref ADJACENT_ATTACK: [[Bitboard; Piece::BPawnX as usize + 1]; SQUARE_NB] = {
        let mut aa: [[Bitboard; Piece::BPawnX as usize + 1]; SQUARE_NB] = [[0; Piece::BPawnX as usize + 1]; SQUARE_NB];

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

                    aa[i][*piece as usize] |= 1 << m.to;
                }
            }
            position.board[i] = Piece::NoPiece;
        }

        return aa;
    };

    /// 飛車の横方向の利きを参照するために用いるmask
    static ref ROOK_MASK1: [u32; SQUARE_NB] = {
        let mut m: [u32; SQUARE_NB] = [0; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let left: usize = (i / 5) * 5;

            for j in 0..5 {
                m[i] |= 1 << (left + j);
            }
        }

        return m;
    };

    /// 飛車の縦方向の利きを参照するために用いるmask
    static ref ROOK_MASK2: [u32; SQUARE_NB] = {
        let mut m: [u32; SQUARE_NB] = [0; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let top: usize = i % 5;

            for j in 0..5 {
                m[i] |= 1 << (top + 5 * j);
            }
        }

        return m;
    };

    /// 飛車の横方向の利きを保持するbitboard
    /// ROOK_ATTACK[pext((player_bb[WHITE] | player_bb[BLACK]), mask)][rook_square]として参照する
    static ref ROOK_ATTACK1: [[Bitboard; 32]; SQUARE_NB] = {
        let mut ba: [[Bitboard; 32]; SQUARE_NB] = [[0; 32]; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let left: usize = (i / 5) * 5;

            for piece_bb in 0..32 {
                let mut position: Position = Position::empty_board();

                for j in 0..5 {
                    if piece_bb & (1 << j) != 0 {
                        position.board[left + j] = Piece::BPawn;
                    }
                }
                position.board[i] = Piece::WRook;

                let moves = position.generate_moves(true, false);

                for m in moves {
                    // 横方向の合法手のみ取りだす
                    if m.to / 5 != i as u8 / 5 {
                        continue;
                    }

                    ba[piece_bb][i] |= 1 << m.to;
                }
            }
        }

        return ba;
    };

    /// 飛車の縦方向の利きを保持するbitboard
    /// ROOK_ATTACK[pext((player_bb[WHITE] | player_bb[BLACK]), mask)][rook_square]として参照する
    static ref ROOK_ATTACK2: [[Bitboard; 32]; SQUARE_NB] = {
        let mut ba: [[Bitboard; 32]; SQUARE_NB] = [[0; 32]; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let top: usize = i % 5;

            for piece_bb in 0..32 {
                let mut position: Position = Position::empty_board();

                for j in 0..5 {
                    if piece_bb & (1 << j) != 0 {
                        position.board[top + 5 * j] = Piece::BPawn;
                    }
                }
                position.board[i] = Piece::WRook;

                let moves = position.generate_moves(true, false);

                for m in moves {
                    // 縦方向の合法手のみ取りだす
                    if m.to % 5 != i as u8 % 5 {
                        continue;
                    }

                    ba[piece_bb][i] |= 1 << m.to;
                }
            }
        }

        return ba;
    };
}

pub fn init() {
    lazy_static::initialize(&ROOK_MASK1);
    lazy_static::initialize(&ROOK_MASK2);

    lazy_static::initialize(&ADJACENT_ATTACK);
    lazy_static::initialize(&ROOK_ATTACK1);
    lazy_static::initialize(&ROOK_ATTACK2);
}

pub fn rook_attack(position: &Position, square: usize) -> Bitboard {
    let piece_bb = position.piece_bb[Color::White as usize] | position.piece_bb[Color::Black as usize];

    ROOK_ATTACK1[piece_bb.pext(ROOK_MASK1[square]) as usize][square] | ROOK_ATTACK2[piece_bb.pext(ROOK_MASK2[square]) as usize][square]
}
