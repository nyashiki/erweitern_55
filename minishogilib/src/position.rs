use rand::seq::SliceRandom;

use pyo3::prelude::*;

use types::*;
use r#move::*;
use bitboard::*;

#[pyclass]
#[derive(Copy, Clone)]
pub struct Position {
    pub side_to_move: Color,
    pub board: [Piece; SQUARE_NB],
    pub hand: [[u8; 5]; 2],

    pub piece_bb: [Bitboard; Piece::BPawnX as usize + 1],
    pub player_bb: [Bitboard; 2],

    pub ply: u16,
    pub kif: [Move; MAX_PLY]
}

#[pymethods]
impl Position {
    #[new]
    pub fn new(obj: &PyRawObject) {
        obj.init(Position {
            side_to_move: Color::NoColor,
            board: [Piece::NoPiece; SQUARE_NB],
            hand: [[0; 5]; 2],
            piece_bb: [0; Piece::BPawnX as usize + 1],
            player_bb: [0; 2],
            ply: 0,
            kif: [NULL_MOVE; MAX_PLY]
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
        self.set_bitboard();
    }

    pub fn generate_moves(self, is_board: bool, is_hand: bool) -> std::vec::Vec<Move> {
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

                    let move_to = ((i as i8) + MOVE_TOS[move_dir as usize]) as u8;

                    let capture_piece = self.board[move_to as usize];

                    // 行き先に自分の駒がある場合には動かせない
                    if capture_piece.get_color() == self.side_to_move {
                        continue;
                    }

                    // 行き場のない歩の不成を禁止
                    if !((self.board[i] == Piece::WPawn && move_to < 5) || (self.board[i] == Piece::BPawn && move_to >= 20)) {
                        moves.push(Move::board_move(self.board[i], i as u8, move_dir, 1, move_to, false, capture_piece));
                    }

                    // 成る手の生成
                    if self.board[i].is_raw() && ((self.side_to_move == Color::White && (move_to < 5 || i < 5)) || (self.side_to_move == Color::Black && (move_to >= 20 || i >= 20))) {
                        moves.push(Move::board_move(self.board[i], i as u8, move_dir, 1, move_to, true, capture_piece));
                    }
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
                            let move_to = ((i as i8) + MOVE_TOS[*move_dir as usize] * (amount as i8)) as u8;

                            let capture_piece = self.board[move_to as usize];

                            // 自分の駒があったらそれ以上進めない
                            if capture_piece.get_color() == self.side_to_move {
                                break;
                            }

                            moves.push(Move::board_move(self.board[i], i as u8, *move_dir, amount, move_to, false, capture_piece));
                            // 成る手の生成
                            if (self.board[i] == Piece::WBishop && (move_to < 5 || i < 5)) || (self.board[i] == Piece::BBishop && (move_to >= 20 || i >= 20)) {
                                moves.push(Move::board_move(self.board[i], i as u8, *move_dir, amount, move_to, true, capture_piece));
                            }

                            // 端まで到達したらそれ以上進めない
                            if move_to / 5 == 0 || move_to / 5 == 4 || move_to % 5 == 0 || move_to % 5 == 4 {
                                break;
                            }

                            // 相手の駒があったらそれ以上進めない
                            if capture_piece.get_color() == self.side_to_move.get_op_color() {
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
                            let move_to = ((i as i8) + MOVE_TOS[*move_dir as usize] * (amount as i8)) as u8;

                            let capture_piece = self.board[move_to as usize];

                            // 自分の駒があったらそれ以上進めない
                            if capture_piece.get_color() == self.side_to_move {
                                break;
                            }

                            moves.push(Move::board_move(self.board[i], i as u8, *move_dir, amount, move_to, false, capture_piece));
                            // 成る手の生成
                            if (self.board[i] == Piece::WRook && (move_to < 5 || i < 5)) || (self.board[i] == Piece::BRook && (move_to >= 20 || i >= 20)) {
                                moves.push(Move::board_move(self.board[i], i as u8, *move_dir, amount, move_to, true, capture_piece));
                            }

                            // 端まで到達したらそれ以上進めない
                            if (*move_dir == Direction::N && move_to / 5 == 0) || (*move_dir == Direction::E && move_to % 5 == 4) || (*move_dir == Direction::S && move_to / 5 == 4) || (*move_dir == Direction::W && move_to % 5 == 0) {
                                break;
                            }

                            // 相手の駒があったらそれ以上進めない
                            if self.board[move_to as usize].get_color() == self.side_to_move.get_op_color() {
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
                if self.hand[self.side_to_move as usize][*piece_type as usize - 2] > 0 {
                    for target in &empty_squares {
                        // 行き場のない駒を打たない
                        if *piece_type == PieceType::Pawn && ((self.side_to_move == Color::White && *target < 5) ||
                                                             (self.side_to_move == Color::Black && *target >= 20)) {
                          continue;
                        }

                        moves.push(Move::hand_move(piece_type.get_piece(self.side_to_move), *target));
                    }
                }
            }
        }

        return moves;
    }

    pub fn make_move(&mut self, m: &Move) {
        if m.amount == 0 {
            // 持ち駒を打つ場合

            self.board[m.to as usize] = m.piece;
            self.hand[self.side_to_move as usize][m.piece.get_piece_type() as usize - 2] -= 1;

            // Bitboardの更新
            self.piece_bb[m.piece as usize] |= 1 << m.to;
            self.player_bb[self.side_to_move as usize] |= 1 << m.to;
        } else {
            // 盤上の駒を動かす場合

            if m.capture_piece != Piece::NoPiece {
                self.hand[self.side_to_move as usize][m.capture_piece.get_piece_type().get_raw() as usize - 2] += 1;

                // Bitboardの更新
                self.piece_bb[m.capture_piece as usize] ^= 1 << m.to;
                self.player_bb[self.side_to_move.get_op_color() as usize] ^= 1 << m.to;
            }

            if m.promotion {
                self.board[m.to as usize] = self.board[m.from as usize].get_promoted();
            } else {
                self.board[m.to as usize] = self.board[m.from as usize];
            }

            self.board[m.from as usize] = Piece::NoPiece;

            // Bitboardの更新
            // 移動先
            self.piece_bb[self.board[m.to as usize] as usize] |= 1 << m.to;
            self.player_bb[self.side_to_move as usize] |= 1 << m.to;
            // 移動元
            self.piece_bb[m.piece as usize] ^= 1 << m.from;
            self.player_bb[self.side_to_move as usize] ^= 1 << m.from;
        }

        // 棋譜に登録
        self.kif[self.ply as usize] = *m;

        // 1手進める
        self.ply += 1;

        // 手番を変える
        self.side_to_move = self.side_to_move.get_op_color();
    }

    pub fn undo_move(&mut self) {
        // 手数を戻す
        let m = self.kif[(self.ply - 1) as usize];
        self.ply -= 1;

        // 手番を戻す
        self.side_to_move = self.side_to_move.get_op_color();

        if m.amount == 0 {
            // 持ち駒を打った場合

            self.board[m.to as usize] = Piece::NoPiece;
            self.hand[self.side_to_move as usize][m.piece.get_piece_type() as usize - 2] += 1;

            // Bitboardのundo
            self.piece_bb[m.piece as usize] ^= 1 << m.to;
            self.player_bb[self.side_to_move as usize] ^= 1 << m.to;
        } else {
            // 盤上の駒を動かした場合

            // Bitboardのundo
            // 移動先
            self.piece_bb[self.board[m.to as usize] as usize] ^= 1 << m.to;
            self.player_bb[self.side_to_move as usize] ^= 1 << m.to;
            // 移動元
            self.piece_bb[m.piece as usize] |= 1 << m.from;
            self.player_bb[self.side_to_move as usize] |= 1 << m.from;

            self.board[m.to as usize] = m.capture_piece;
            self.board[m.from as usize] = m.piece;

            // 相手の駒を取っていた場合には、持ち駒から減らす
            if m.capture_piece != Piece::NoPiece {
              self.hand[self.side_to_move as usize][m.capture_piece.get_piece_type().get_raw() as usize - 2] -= 1;

              // Bitboardのundo
              self.piece_bb[m.capture_piece as usize] |= 1 << m.to;
              self.player_bb[self.side_to_move.get_op_color() as usize] |= 1 << m.to;
            }
        }
    }

    /// 盤上の駒からbitboardを設定する
    fn set_bitboard(&mut self) {
        // 初期化
        for i in 0..Piece::BPawnX as usize + 1 {
            self.piece_bb[i] = 0
        }
        self.player_bb[Color::White as usize] = 0;
        self.player_bb[Color::Black as usize] = 0;

        // 盤上の駒に対応する場所のbitを立てる
        for i in 0..SQUARE_NB {
            if self.board[i] != Piece::NoPiece {
                self.piece_bb[self.board[i] as usize] |= 1 << i;
                self.player_bb[self.board[i].get_color() as usize] |= 1 << i;
            }
        }
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

#[test]
fn move_do_undo_test() {
    const LOOP_NUM: i32 = 10000;

    let mut position = Position {
        side_to_move: Color::NoColor,
        board: [Piece::NoPiece; SQUARE_NB],
        hand: [[0; 5]; 2],
        piece_bb: [0; Piece::BPawnX as usize + 1],
        player_bb: [0; 2],
        ply: 0,
        kif: [NULL_MOVE; MAX_PLY]
    };

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        loop {
            let moves = position.generate_moves(true, true);

            for m in &moves {
                let mut temp_position = position;
                temp_position.make_move(m);
                temp_position.undo_move();

                // make_move -> undo_moveで元の局面と一致するはず
                assert!(position.side_to_move == temp_position.side_to_move);
                for i in 0..SQUARE_NB {
                    assert!(position.board[i] == temp_position.board[i]);
                }
                for i in 0..2 {
                    for j in 0..5 {
                        assert!(position.hand[i][j] == temp_position.hand[i][j]);
                    }
                }
                for i in 0..Piece::BPawnX as usize + 1 {
                    assert!(position.piece_bb[i] == temp_position.piece_bb[i]);
                }
                for i in 0..2 {
                    assert!(position.player_bb[i] == temp_position.player_bb[i]);
                }
                assert!(position.ply == temp_position.ply);
                for i in 0..MAX_PLY {
                    assert!(position.kif[i] == temp_position.kif[i]);
                }
            }

            // ランダムに局面を進める
            let random_move = moves.choose(&mut rng).unwrap();
            if random_move.capture_piece.get_piece_type() == PieceType::King {
                break;
            }
            position.make_move(random_move);
        }
    }
}
