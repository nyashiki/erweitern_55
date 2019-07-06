use bitintr::Pext;

use types::*;
use position::*;

pub type Bitboard = u32;

lazy_static! {
    /// 近接の利きを保持するbitboard
    /// ADJACENT_ATTACK[square][piece]として参照する
    static ref ADJACENT_ATTACK: [[Bitboard; Piece::BPawnX as usize + 1]; SQUARE_NB] = {
        let mut aa: [[Bitboard; Piece::BPawnX as usize + 1]; SQUARE_NB] = [[0; Piece::BPawnX as usize + 1]; SQUARE_NB];

        let mut position: Position = Position::empty_board();

        for i in 0..SQUARE_NB {
            for piece in PIECE_ALL.iter() {
                position.board[i] = *piece;
                position.side_to_move = piece.get_color();

                let moves = position.generate_moves_with_option(true, false, true);

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

    /// 角の右上--左下方向の利きを参照するために用いるmask
    static ref BISHOP_MASK1: [u32; SQUARE_NB] = {
        let mut m: [u32; SQUARE_NB] = [0; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let right_top = {
                let mut y = i / 5;
                let mut x = i % 5;

                while y > 0 && x < 4 {
                    y -= 1;
                    x += 1;
                }

                5 * y + x
            };

            let left_bottom = {
                let mut y = i / 5;
                let mut x = i % 5;

                while y < 4 && x > 0 {
                    y += 1;
                    x -= 1;
                }

                5 * y + x
            };

            let mut square = right_top;
            loop {
                m[i] |= 1 << square;
                if square == left_bottom {
                    break;
                }
                square += 4;
            }
        }

        return m;
    };

    /// 角の左上--右下方向の利きを参照するために用いるmask
    static ref BISHOP_MASK2: [u32; SQUARE_NB] = {
        let mut m: [u32; SQUARE_NB] = [0; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let left_top = {
                let mut y = i / 5;
                let mut x = i % 5;

                while y > 0 && x > 0 {
                    y -= 1;
                    x -= 1;
                }

                5 * y + x
            };

            let right_bottom = {
                let mut y = i / 5;
                let mut x = i % 5;

                while y < 4 && x < 4 {
                    y += 1;
                    x += 1;
                }

                5 * y + x
            };

            let mut square = left_top;
            loop {
                m[i] |= 1 << square;
                if square == right_bottom {
                    break;
                }
                square += 6;
            }
        }

        return m;
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

    /// 角の右上--左下方向の利きを保持するbitboard
    /// BISHOP_ATTACK1[bishop_square][pext((player_bb[WHITE] | player_bb[BLACK]), mask)]として参照する
    static ref BISHOP_ATTACK1: [[Bitboard; 32]; SQUARE_NB] = {
        let mut ba: [[Bitboard; 32]; SQUARE_NB] = [[0; 32]; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let right_top = {
                let mut y = i / 5;
                let mut x = i % 5;

                while y > 0 && x < 4 {
                    y -= 1;
                    x += 1;
                }

                5 * y + x
            };

            let left_bottom = {
                let mut y = i / 5;
                let mut x = i % 5;

                while y < 4 && x > 0 {
                    y += 1;
                    x -= 1;
                }

                5 * y + x
            };

            for player_bb in 0..32 {
                let mut position: Position = Position::empty_board();

                for j in 0..5 {
                    if player_bb & (1 << j) != 0 {
                        position.board[right_top + 4 * j] = Piece::BPawn;
                    }
                    if right_top + 4 * j == left_bottom {
                        break;
                    }
                }
                position.board[i] = Piece::WBishop;
                position.side_to_move = Color::White;

                let moves = position.generate_moves_with_option(true, false, true);

                for m in moves {
                    // 右上, 左下方向の合法手のみ取りだす
                    if m.direction == Direction::NE || m.direction == Direction::SW {
                        ba[i][player_bb] |= 1 << m.to;
                    }
                }
            }
        }

        return ba;
    };

    /// 角の左上--右下方向の利きを保持するbitboard
    /// BISHOP_ATTACK2[bishop_square][pext((player_bb[WHITE] | player_bb[BLACK]), mask)]として参照する
    static ref BISHOP_ATTACK2: [[Bitboard; 32]; SQUARE_NB] = {
        let mut ba: [[Bitboard; 32]; SQUARE_NB] = [[0; 32]; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let left_top = {
                let mut y = i / 5;
                let mut x = i % 5;

                while y > 0 && x > 0 {
                    y -= 1;
                    x -= 1;
                }

                5 * y + x
            };

            let right_bottom = {
                let mut y = i / 5;
                let mut x = i % 5;

                while y < 4 && x < 4 {
                    y += 1;
                    x += 1;
                }

                5 * y + x
            };

            for player_bb in 0..32 {
                let mut position: Position = Position::empty_board();

                for j in 0..5 {
                    if player_bb & (1 << j) != 0 {
                        position.board[left_top + 6 * j] = Piece::BPawn;
                    }
                    if left_top + 6 * j == right_bottom {
                        break;
                    }
                }
                position.board[i] = Piece::WBishop;
                position.side_to_move = Color::White;

                let moves = position.generate_moves_with_option(true, false, true);

                for m in moves {
                    // 左上--右下方向の合法手のみ取りだす
                    if m.direction == Direction::NW || m.direction == Direction::SE {
                        ba[i][player_bb] |= 1 << m.to;
                    }
                }
            }
        }

        return ba;
    };

    /// 飛車の横方向の利きを保持するbitboard
    /// ROOK_ATTACK1[rook_square][pext((player_bb[WHITE] | player_bb[BLACK]), mask)]として参照する
    static ref ROOK_ATTACK1: [[Bitboard; 32]; SQUARE_NB] = {
        let mut ra: [[Bitboard; 32]; SQUARE_NB] = [[0; 32]; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let left: usize = (i / 5) * 5;

            for player_bb in 0..32 {
                let mut position: Position = Position::empty_board();

                for j in 0..5 {
                    if player_bb & (1 << j) != 0 {
                        position.board[left + j] = Piece::BPawn;
                    }
                }
                position.board[i] = Piece::WRook;
                position.side_to_move = Color::White;

                let moves = position.generate_moves_with_option(true, false, true);

                for m in moves {
                    // 横方向の合法手のみ取りだす
                    if m.direction == Direction::E || m.direction == Direction::W {
                        ra[i][player_bb] |= 1 << m.to;
                    }
                }
            }
        }

        return ra;
    };

    /// 飛車の縦方向の利きを保持するbitboard
    /// ROOK_ATTACK2[rook_square][pext((player_bb[WHITE] | player_bb[BLACK]), mask)]として参照する
    static ref ROOK_ATTACK2: [[Bitboard; 32]; SQUARE_NB] = {
        let mut ra: [[Bitboard; 32]; SQUARE_NB] = [[0; 32]; SQUARE_NB];

        for i in 0..SQUARE_NB {
            let top: usize = i % 5;

            for player_bb in 0..32 {
                let mut position: Position = Position::empty_board();

                for j in 0..5 {
                    if player_bb & (1 << j) != 0 {
                        position.board[top + 5 * j] = Piece::BPawn;
                    }
                }
                position.board[i] = Piece::WRook;
                position.side_to_move = Color::White;

                let moves = position.generate_moves_with_option(true, false, true);

                for m in moves {
                    // 縦方向の合法手のみ取りだす
                    if m.direction == Direction::N || m.direction == Direction::S {
                        ra[i][player_bb] |= 1 << m.to;
                    }
                }
            }
        }

        return ra;
    };
}

pub fn init() {
    lazy_static::initialize(&BISHOP_MASK1);
    lazy_static::initialize(&BISHOP_MASK2);
    lazy_static::initialize(&ROOK_MASK1);
    lazy_static::initialize(&ROOK_MASK2);

    lazy_static::initialize(&ADJACENT_ATTACK);
    lazy_static::initialize(&BISHOP_ATTACK1);
    lazy_static::initialize(&BISHOP_ATTACK2);
    lazy_static::initialize(&ROOK_ATTACK1);
    lazy_static::initialize(&ROOK_ATTACK2);
}

pub fn adjacent_attack(square: usize, piece: Piece) -> Bitboard {
    ADJACENT_ATTACK[square][piece as usize]
}

pub fn bishop_attack(square: usize, player_bb: Bitboard) -> Bitboard {
    BISHOP_ATTACK1[square][player_bb.pext(BISHOP_MASK1[square]) as usize] | BISHOP_ATTACK2[square][player_bb.pext(BISHOP_MASK2[square]) as usize]
}

pub fn rook_attack(square: usize, player_bb: Bitboard) -> Bitboard {
    ROOK_ATTACK1[square][player_bb.pext(ROOK_MASK1[square]) as usize] | ROOK_ATTACK2[square][player_bb.pext(ROOK_MASK2[square]) as usize]
}

/// 一番末尾の1の場所を返す
pub fn get_square(bb: Bitboard) -> usize {
    bb.trailing_zeros() as usize
}
