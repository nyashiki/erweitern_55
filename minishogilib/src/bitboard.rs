use bitintr::Pext;
use bitintr::Popcnt;

use position::*;
use types::*;

pub type Bitboard = u32;

lazy_static! {
    /// 近接の利きを保持するbitboard
    /// ADJACENT_ATTACK[square][piece]として参照する
    static ref ADJACENT_ATTACK: [[Bitboard; Piece::BPawnX as usize + 1]; SQUARE_NB] = {
        let mut aa: [[Bitboard; Piece::BPawnX as usize + 1]; SQUARE_NB] = [[0; Piece::BPawnX as usize + 1]; SQUARE_NB];

        let mut position: Position = Position::empty_board();

        const MOVE_TOS: [i8; 8] = [-5, -4, 1, 6, 5, 4, -1, -6];

        for i in 0..SQUARE_NB {
            for piece in PIECE_ALL.iter() {
                position.board[i] = *piece;
                position.side_to_move = piece.get_color();

                for move_dir in position.board[i].get_move_dirs() {
                    // これ以上左に行けない
                    if i % 5 == 0
                        && (move_dir == Direction::SW
                            || move_dir == Direction::W
                            || move_dir == Direction::NW)
                    {
                        continue;
                    }

                    // これ以上上に行けない
                    if i / 5 == 0
                        && (move_dir == Direction::N
                            || move_dir == Direction::NE
                            || move_dir == Direction::NW)
                    {
                        continue;
                    }

                    // これ以上右に行けない
                    if i % 5 == 4
                        && (move_dir == Direction::NE
                            || move_dir == Direction::E
                            || move_dir == Direction::SE)
                    {
                        continue;
                    }

                    // これ以上下に行けない
                    if i / 5 == 4
                        && (move_dir == Direction::SE
                            || move_dir == Direction::S
                            || move_dir == Direction::SW)
                    {
                        continue;
                    }

                    let move_to = ((i as i8) + MOVE_TOS[move_dir as usize]) as usize;
                    aa[i][*piece as usize] |= 1 << move_to;
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

                const MOVE_TOS: [i8; 8] = [-5, -4, 1, 6, 5, 4, -1, -6];
                const MOVE_DIRS: [Direction; 2] = [Direction::NE, Direction::SW];
                for move_dir in &MOVE_DIRS {
                    // これ以上左に行けない
                    if i % 5 == 0 && (*move_dir == Direction::SW || *move_dir == Direction::NW)
                    {
                        continue;
                    }

                    // これ以上上に行けない
                    if i / 5 == 0 && (*move_dir == Direction::NE || *move_dir == Direction::NW)
                    {
                        continue;
                    }

                    // これ以上右に行けない
                    if i % 5 == 4 && (*move_dir == Direction::NE || *move_dir == Direction::SE)
                    {
                        continue;
                    }

                    // これ以上下に行けない
                    if i / 5 == 4 && (*move_dir == Direction::SE || *move_dir == Direction::SW)
                    {
                        continue;
                    }

                    for amount in 1..5 {
                        let move_to = ((i as i8)
                            + MOVE_TOS[*move_dir as usize] * (amount as i8))
                            as usize;

                        ba[i][player_bb] |= 1 << move_to;

                        if position.board[move_to] != Piece::NoPiece {
                            break;
                        }

                        // 端まで到達したらそれ以上進めない
                        if move_to / 5 == 0
                            || move_to / 5 == 4
                            || move_to % 5 == 0
                            || move_to % 5 == 4
                        {
                            break;
                        }
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

                const MOVE_TOS: [i8; 8] = [-5, -4, 1, 6, 5, 4, -1, -6];
                const MOVE_DIRS: [Direction; 2] = [Direction::NW, Direction::SE];
                for move_dir in &MOVE_DIRS {
                    // これ以上左に行けない
                    if i % 5 == 0 && (*move_dir == Direction::SW || *move_dir == Direction::NW)
                    {
                        continue;
                    }

                    // これ以上上に行けない
                    if i / 5 == 0 && (*move_dir == Direction::NE || *move_dir == Direction::NW)
                    {
                        continue;
                    }

                    // これ以上右に行けない
                    if i % 5 == 4 && (*move_dir == Direction::NE || *move_dir == Direction::SE)
                    {
                        continue;
                    }

                    // これ以上下に行けない
                    if i / 5 == 4 && (*move_dir == Direction::SE || *move_dir == Direction::SW)
                    {
                        continue;
                    }

                    for amount in 1..5 {
                        let move_to = ((i as i8)
                            + MOVE_TOS[*move_dir as usize] * (amount as i8))
                            as usize;

                        ba[i][player_bb] |= 1 << move_to;

                        if position.board[move_to] != Piece::NoPiece {
                            break;
                        }

                        // 端まで到達したらそれ以上進めない
                        if move_to / 5 == 0
                            || move_to / 5 == 4
                            || move_to % 5 == 0
                            || move_to % 5 == 4
                        {
                            break;
                        }
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

                const MOVE_TOS: [i8; 8] = [-5, -4, 1, 6, 5, 4, -1, -6];
                const MOVE_DIRS: [Direction; 2] = [Direction::E, Direction::W];
                for move_dir in &MOVE_DIRS {
                    // これ以上左に行けない
                    if i % 5 == 0 && *move_dir == Direction::W
                    {
                        continue;
                    }
                    // これ以上右に行けない
                    if i % 5 == 4 && *move_dir == Direction::E
                    {
                        continue;
                    }

                    for amount in 1..5 {
                        let move_to = ((i as i8)
                            + MOVE_TOS[*move_dir as usize] * (amount as i8))
                            as usize;

                        ra[i][player_bb] |= 1 << move_to;

                        if position.board[move_to] != Piece::NoPiece {
                            break;
                        }

                        // 端まで到達したらそれ以上進めない
                        if (*move_dir == Direction::E && move_to % 5 == 4) || (*move_dir == Direction::W && move_to % 5 == 0)
                        {
                            break;
                        }
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

                const MOVE_TOS: [i8; 8] = [-5, -4, 1, 6, 5, 4, -1, -6];
                const MOVE_DIRS: [Direction; 2] = [Direction::N, Direction::S];
                for move_dir in &MOVE_DIRS {
                    // これ以上上に行けない
                    if i / 5 == 0 && *move_dir == Direction::N
                    {
                        continue;
                    }

                    // これ以上下に行けない
                    if i / 5 == 4 && *move_dir == Direction::S
                    {
                        continue;
                    }

                    for amount in 1..5 {
                        let move_to = ((i as i8)
                            + MOVE_TOS[*move_dir as usize] * (amount as i8))
                            as usize;

                        ra[i][player_bb] |= 1 << move_to;

                        if position.board[move_to] != Piece::NoPiece {
                            break;
                        }

                        // 端まで到達したらそれ以上進めない
                        if (*move_dir == Direction::N && move_to / 5 == 0) || (*move_dir == Direction::S && move_to / 5 == 4)
                        {
                            break;
                        }
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
    BISHOP_ATTACK1[square][player_bb.pext(BISHOP_MASK1[square]) as usize]
        | BISHOP_ATTACK2[square][player_bb.pext(BISHOP_MASK2[square]) as usize]
}

pub fn rook_attack(square: usize, player_bb: Bitboard) -> Bitboard {
    ROOK_ATTACK1[square][player_bb.pext(ROOK_MASK1[square]) as usize]
        | ROOK_ATTACK2[square][player_bb.pext(ROOK_MASK2[square]) as usize]
}

/// 一番末尾の1の場所を返す
pub fn get_square(bb: Bitboard) -> usize {
    bb.trailing_zeros() as usize
}

/// 1の数を返す
pub fn get_counts(bb: Bitboard) -> u32 {
    bb.popcnt()
}
