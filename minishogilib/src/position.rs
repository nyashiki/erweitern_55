use pyo3::prelude::*;

use types::*;
use r#move::*;

#[pyclass]
#[derive(Copy, Clone)]
pub struct Position {
    pub side_to_move: Color,
    pub board: [Piece; SQUARE_NB],
    pub hand: [[u8; 5]; 2],
    pub ply: u16
}

#[pymethods]
impl Position {
    #[new]
    pub fn new(obj: &PyRawObject) {
        obj.init(Position {
            side_to_move: Color::NoColor,
            board: [Piece::NoPiece; SQUARE_NB],
            hand: [[0; 5]; 2],
            ply: 0
        });
    }

    pub fn print(self) {
        println!("side_to_move: {:?}", self.side_to_move);

        for y in 0..5 {
            for x in 0..5 {
                print!("{}", self.board[y * 5 + x]);
            }
            println!("");
        }

        let hand_str = ["G", "S", "B", "R", "P"];

        print!("WHITE HAND: ");
        for i in 0..5 {
            print!("{}: {}, ", hand_str[i], self.hand[(Color::White as usize)][i]);
        }
        println!("");

        print!("BLACK HAND: ");
        for i in 0..5 {
            print!("{}: {}, ", hand_str[i], self.hand[(Color::Black as usize)][i]);
        }
        println!("");

        println!("ply: {}", self.ply);
    }

    pub fn set_sfen(&mut self, sfen: &str) {
        let mut square: usize = 0;
        let mut promote: bool = false;

        let mut sfen_split = sfen.split_whitespace();

        // sfenから盤面を設定
        for c in sfen_split.next().unwrap().chars() {
            if c == '+' {
                promote = true;
                continue;
            }

            if c == '/' {
                continue;
            }

            if c.is_ascii_digit() {
                square += ((c as u8) - ('0' as u8)) as usize;
                continue;
            }

            let mut piece = char_to_piece(c);

            if promote {
                piece = piece.get_promoted();
            }

            self.board[square] = piece;

            promote = false;
            square += 1;
        }

        // 手番を設定
        if sfen_split.next() == Some("b") {
            self.side_to_move = Color::White;
        } else {
            self.side_to_move = Color::Black;
        }

        // 持ち駒を設定
        let mut count: u8 = 1;
        for c in sfen_split.next().unwrap().chars() {
            if c == '-' {
                continue;
            }

            if c.is_ascii_digit() {
                count = (c as u8) - ('0' as u8);
                continue;
            }

            let piece = char_to_piece(c);
            let color = piece.get_color();
            let piece_type = piece.get_piece_type();
            let hand_index = (piece_type as usize) - 2;

            self.hand[color as usize][hand_index] = count;

            count = 1;
        }
    }

    pub fn set_start_position(&mut self) {
        static START_POSITION_SFEN: &str = "rbsgk/4p/5/P4/KGSBR b - 1";

        self.set_sfen(START_POSITION_SFEN);
    }

    pub fn generate_moves(self, is_board: bool, is_hand: bool) -> std::vec::Vec<Move> {
        // ToDo: 駒が成る処理

        let mut moves: Vec<Move> = Vec::new();

        if is_board {

            for i in 0..SQUARE_NB {
                if self.board[i].get_color() != self.side_to_move {
                    continue;
                }

                const MOVE_TOS: [i8; 8] = [-5, -4, 1, 6, 5, 4, -1, -6];

                // 飛び駒以外の駒の移動
                for move_dir in self.board[i].get_move_dirs() {
                    // これ以上左に行けない
                    if i % 5 == 0 && (move_dir == Direction::SW || move_dir == Direction::W || move_dir == Direction::NW) {
                        continue;
                    }

                    // これ以上上に行けない
                    if i / 5 == 0 && (move_dir == Direction::N || move_dir == Direction::NE || move_dir == Direction::NW) {
                        continue;
                    }

                    // これ以上右に行けない
                    if i % 5 == 4 && (move_dir == Direction::NE || move_dir == Direction::E || move_dir == Direction::SE) {
                        continue;
                    }

                    // これ以上下に行けない
                    if i / 5 == 4 && (move_dir == Direction::SE || move_dir == Direction::S || move_dir == Direction::SW) {
                        continue;
                    }

                    let move_to = ((i as i8) + MOVE_TOS[move_dir as usize]) as usize;
                    // 行き先に自分の駒がある場合には動かせない
                    if self.board[move_to].get_color() == self.side_to_move {
                        continue;
                    }

                    moves.push(Move::board_move(self.board[i], i as u8, move_dir, 1, false));
                }

                // 飛び駒の移動
                // 角、馬
                if self.board[i].get_piece_type() == PieceType::Bishop || self.board[i].get_piece_type() == PieceType::BishopX {
                    const MOVE_DIRS: [Direction; 4] = [Direction::NE, Direction::SE, Direction::SW, Direction::NW];

                    for move_dir in &MOVE_DIRS {
                        // これ以上左に行けない
                        if i % 5 == 0 && (*move_dir == Direction::SW || *move_dir == Direction::NW) {
                            continue;
                        }

                        // これ以上上に行けない
                        if i / 5 == 0 && (*move_dir == Direction::NE || *move_dir == Direction::NW) {
                            continue;
                        }

                        // これ以上右に行けない
                        if i % 5 == 4 && (*move_dir == Direction::NE || *move_dir == Direction::SE) {
                            continue;
                        }

                        // これ以上下に行けない
                        if i / 5 == 4 && (*move_dir == Direction::SE || *move_dir == Direction::SW) {
                            continue;
                        }

                        for amount in 1..5 {
                            let move_to = ((i as i8) + MOVE_TOS[*move_dir as usize] * (amount as i8)) as usize;

                            // 自分の駒があったらそれ以上進めない
                            if self.board[move_to].get_color() == self.side_to_move {
                                break;
                            }

                            moves.push(Move::board_move(self.board[i], i as u8, *move_dir, amount, false));

                            // 端まで到達したらそれ以上進めない
                            if move_to / 5 == 0 || move_to / 5 == 4 || move_to % 5 == 0 || move_to % 5 == 4 {
                                break;
                            }

                            // 相手の駒があったらそれ以上進めない
                            if self.board[move_to].get_color() == self.side_to_move.get_op_color() {
                                break;
                            }
                        }
                    }
                }

                // 飛、龍
                if self.board[i].get_piece_type() == PieceType::Rook || self.board[i].get_piece_type() == PieceType::RookX {
                    const MOVE_DIRS: [Direction; 4] = [Direction::N, Direction::E, Direction::S, Direction::W];

                    for move_dir in &MOVE_DIRS {
                        // これ以上左に行けない
                        if i % 5 == 0 && *move_dir == Direction::W {
                            continue;
                        }

                        // これ以上上に行けない
                        if i / 5 == 0 && *move_dir == Direction::N {
                            continue;
                        }

                        // これ以上右に行けない
                        if i % 5 == 4 && *move_dir == Direction::E {
                            continue;
                        }

                        // これ以上下に行けない
                        if i / 5 == 4 && *move_dir == Direction::S {
                            continue;
                        }

                        for amount in 1..5 {
                            let move_to = ((i as i8) + MOVE_TOS[*move_dir as usize] * (amount as i8)) as usize;

                            // 自分の駒があったらそれ以上進めない
                            if self.board[move_to].get_color() == self.side_to_move {
                                break;
                            }

                            moves.push(Move::board_move(self.board[i], i as u8, *move_dir, amount, false));

                            // 端まで到達したらそれ以上進めない
                            if (*move_dir == Direction::N && move_to / 5 == 0) || (*move_dir == Direction::E && move_to % 5 == 4) || (*move_dir == Direction::S && move_to / 5 == 4) || (*move_dir == Direction::W && move_to % 5 == 0) {
                                break;
                            }

                            // 相手の駒があったらそれ以上進めない
                            if self.board[move_to].get_color() == self.side_to_move.get_op_color() {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if is_hand {
            // 駒のない升を列挙
            let mut empty_squares: Vec<u8> = Vec::new();
            for i in 0..SQUARE_NB {
                if self.board[i] == Piece::NoPiece {
                    empty_squares.push(i as u8);
                }
            }

            for piece_type in HAND_PIECE_TYPE_ALL.iter() {
                if self.hand[self.side_to_move as usize][*piece_type as usize] > 0 {
                    for target in &empty_squares {
                        moves.push(Move::hand_move(piece_type.get_piece(self.side_to_move), *target));
                    }
                }
            }
        }

        return moves;
    }
}

fn char_to_piece(c: char) -> Piece {
    match c {
        'K' => Piece::WKing,
        'G' => Piece::WGold,
        'S' => Piece::WSilver,
        'B' => Piece::WBishop,
        'R' => Piece::WRook,
        'P' => Piece::WPawn,

        'k' => Piece::BKing,
        'g' => Piece::BGold,
        's' => Piece::BSilver,
        'b' => Piece::BBishop,
        'r' => Piece::BRook,
        'p' => Piece::BPawn,

        _ => Piece::NoPiece
    }
}
