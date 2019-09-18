#[cfg(test)]
use rand::seq::SliceRandom;

use pyo3::prelude::*;

use bitboard::*;
use r#move::*;
use types::*;

#[pyclass]
#[derive(Copy, Clone)]
pub struct Position {
    pub side_to_move: Color,
    pub board: [Piece; SQUARE_NB],
    pub hand: [[u8; 5]; 2],
    pub pawn_flags: [u8; 2],

    pub piece_bb: [Bitboard; Piece::BPawnX as usize + 1],
    pub player_bb: [Bitboard; 2],

    pub ply: u16,
    pub kif: [Move; MAX_PLY + 1],

    pub hash: [u64; MAX_PLY + 1],

    pub adjacent_check_bb: [Bitboard; MAX_PLY + 1], // 近接駒による王手を表すbitboard
    pub long_check_bb: [Bitboard; MAX_PLY + 1],     /* 長い利きを持つ駒による王手を表すbitboard */

    pub sequent_check_count: [[i8; 2]; MAX_PLY + 1],
}

#[pymethods]
impl Position {
    #[new]
    pub fn new(obj: &PyRawObject) {
        obj.init(Position::empty_board());
    }

    /// entireがTrueの時には，過去の棋譜もコピーする
    pub fn copy(&self, entire: bool) -> Position {
        if entire {
            return *self;
        }

        let mut position = Position::empty_board();
        position.side_to_move = self.side_to_move;
        for i in 0..SQUARE_NB {
            position.board[i] = self.board[i]
        }
        for i in 0..2 {
            for j in 0..5 {
                position.hand[i][j] = self.hand[i][j];
            }
            position.pawn_flags[i] = self.pawn_flags[i];
            position.player_bb[i] = self.player_bb[i];
        }

        for piece in &PIECE_ALL {
            position.piece_bb[*piece as usize] = self.piece_bb[*piece as usize];
        }

        position.hash[0] = self.hash[self.ply as usize];
        position.adjacent_check_bb[0] = self.adjacent_check_bb[self.ply as usize];
        position.long_check_bb[0] = self.long_check_bb[self.ply as usize];
        position.sequent_check_count[0] = self.sequent_check_count[self.ply as usize];

        return position;
    }

    pub fn print(&self) {
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

        println!("hash: {:x}", self.get_hash());
        println!("repetition: {}", self.get_repetition());
    }

    pub fn sfen(&self, history: bool) -> String {
        if history {
            let mut position = *self;

            for _ in 0..self.ply {
                position.undo_move();
            }

            let mut sfen_position = position.get_sfen_position();

            if self.ply > 0 {
                sfen_position.push_str(" moves");
            }

            for i in 0..self.ply {
                sfen_position.push_str(&format!(" {}", self.kif[i as usize].sfen()));
            }

            return sfen_position;
        } else {
            return self.get_sfen_position();
        }
    }

    pub fn get_kif(&self) -> std::vec::Vec<String> {
        self.kif[0..self.ply as usize].to_vec().into_iter().map(|x| x.sfen()).collect()
    }

    pub fn get_csa_kif(&self) -> std::vec::Vec<String> {
        self.kif[0..self.ply as usize].to_vec().into_iter().map(|x| x.csa_sfen()).collect()
    }

    pub fn set_sfen(&mut self, sfen: &str) {
        // 初期化
        for i in 0..SQUARE_NB {
            self.board[i] = Piece::NoPiece;
        }
        for i in 0..2 {
            for j in 0..5 {
                self.hand[i][j] = 0;
            }

            self.pawn_flags[i] = 0;
        }

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

            if piece == Piece::WPawn {
                self.pawn_flags[Color::White as usize] |= 1 << (square % 5);
            } else if piece == Piece::BPawn {
                self.pawn_flags[Color::Black as usize] |= 1 << (square % 5);
            }

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

        self.set_bitboard();
        self.set_check_bb();
        self.hash[0] = self.calculate_hash();

        self.ply = 0;

        sfen_split.next(); // sfenプロトコルで常に1が格納されているはずなので、読み飛ばす

        if sfen_split.next() == Some("moves") {
            loop {
                let sfen_move = sfen_split.next();

                if !sfen_move.is_some() {
                    break;
                }

                let m = self.sfen_to_move(sfen_move.unwrap().to_string());
                self.do_move(&m);
            }
        }
    }

    pub fn set_start_position(&mut self) {
        static START_POSITION_SFEN: &str = "rbsgk/4p/5/P4/KGSBR b - 1";

        self.set_sfen(START_POSITION_SFEN);
    }

    /// sfen形式での指し手をMove構造体に変換する
    pub fn sfen_to_move(&self, sfen: String) -> Move {
        if sfen.as_bytes()[1] as char == '*' {
            let piece = char_to_piece(sfen.as_bytes()[0] as char)
                .get_piece_type()
                .get_piece(self.side_to_move);
            let to = sfen_to_square(sfen[2..4].to_string());

            Move::hand_move(piece, to)
        } else {
            let from = sfen_to_square(sfen[0..2].to_string());
            let to = sfen_to_square(sfen[2..4].to_string());
            let promotion = sfen.len() == 5;
            let piece = self.board[from];
            let (direction, amount) = get_relation(from, to);
            let capture_piece = self.board[to];

            Move::board_move(piece, from, direction, amount, to, promotion, capture_piece)
        }
    }

    pub fn get_side_to_move(&self) -> usize {
        return self.side_to_move as usize;
    }

    pub fn generate_moves(&self) -> std::vec::Vec<Move> {
        return self.generate_moves_with_option(true, true, false);
    }

    pub fn do_move(&mut self, m: &Move) {
        assert!(m.capture_piece.get_piece_type() != PieceType::King);

        self.hash[self.ply as usize + 1] = self.hash[self.ply as usize];

        if m.amount == 0 {
            // 持ち駒を打つ場合

            self.board[m.to as usize] = m.piece;
            self.hand[self.side_to_move as usize][m.piece.get_piece_type() as usize - 2] -= 1;

            // Bitboardの更新
            self.piece_bb[m.piece as usize] |= 1 << m.to;
            self.player_bb[self.side_to_move as usize] |= 1 << m.to;

            // 二歩フラグの更新
            if m.piece.get_piece_type() == PieceType::Pawn {
                self.pawn_flags[self.side_to_move as usize] |= 1 << (m.to % 5);
            }

            // hash値の更新
            self.hash[self.ply as usize + 1] ^= ::zobrist::BOARD_TABLE[m.to][m.piece as usize];
        } else {
            // 盤上の駒を動かす場合

            if m.capture_piece != Piece::NoPiece {
                self.hand[self.side_to_move as usize]
                    [m.capture_piece.get_piece_type().get_raw() as usize - 2] += 1;

                // Bitboardの更新
                self.piece_bb[m.capture_piece as usize] ^= 1 << m.to;
                self.player_bb[self.side_to_move.get_op_color() as usize] ^= 1 << m.to;

                // 二歩フラグの更新
                if m.capture_piece.get_piece_type() == PieceType::Pawn {
                    self.pawn_flags[self.side_to_move.get_op_color() as usize] ^= 1 << (m.to % 5);
                }

                // hashの更新
                self.hash[self.ply as usize + 1] ^=
                    ::zobrist::BOARD_TABLE[m.to][m.capture_piece as usize];
            }

            if m.promotion {
                self.board[m.to as usize] = m.piece.get_promoted();
                // 二歩フラグの更新
                if m.piece.get_piece_type() == PieceType::Pawn {
                    self.pawn_flags[self.side_to_move as usize] ^= 1 << (m.to % 5);
                }
            } else {
                self.board[m.to as usize] = m.piece;
            }

            self.board[m.from as usize] = Piece::NoPiece;

            // Bitboardの更新
            // 移動先
            self.piece_bb[self.board[m.to as usize] as usize] |= 1 << m.to;
            self.player_bb[self.side_to_move as usize] |= 1 << m.to;
            // 移動元
            self.piece_bb[m.piece as usize] ^= 1 << m.from;
            self.player_bb[self.side_to_move as usize] ^= 1 << m.from;

            // hash値の更新
            self.hash[self.ply as usize + 1] ^= ::zobrist::BOARD_TABLE[m.from][m.piece as usize];
            self.hash[self.ply as usize + 1] ^=
                ::zobrist::BOARD_TABLE[m.to][self.board[m.to] as usize];
        }

        self.hash[self.ply as usize + 1] ^= 1; // 手番bitの反転

        // 棋譜に登録
        self.kif[self.ply as usize] = *m;

        // 1手進める
        self.ply += 1;

        // 手番を変える
        self.side_to_move = self.side_to_move.get_op_color();

        // 王手している駒を記録
        self.set_check_bb();

        // 連続王手のカウント
        if self.adjacent_check_bb[self.ply as usize] != 0
            || self.long_check_bb[self.ply as usize] != 0
        {
            self.sequent_check_count[self.ply as usize]
                [self.side_to_move.get_op_color() as usize] = self.sequent_check_count
                [self.ply as usize - 1][self.side_to_move.get_op_color() as usize]
                + 1;
        } else {
            self.sequent_check_count[self.ply as usize]
                [self.side_to_move.get_op_color() as usize] = 0;
        }
        self.sequent_check_count[self.ply as usize][self.side_to_move as usize] =
            self.sequent_check_count[self.ply as usize - 1][self.side_to_move as usize];
    }

    pub fn undo_move(&mut self) {
        assert!(self.ply > 0);

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

            // 二歩フラグのundo
            if m.piece.get_piece_type() == PieceType::Pawn {
                self.pawn_flags[self.side_to_move as usize] ^= 1 << (m.to % 5);
            }
        } else {
            // 盤上の駒を動かした場合

            // Bitboardのundo
            // 移動先
            assert!(self.board[m.to as usize] != Piece::NoPiece);
            self.piece_bb[self.board[m.to as usize] as usize] ^= 1 << m.to;
            self.player_bb[self.side_to_move as usize] ^= 1 << m.to;
            // 移動元
            self.piece_bb[m.piece as usize] |= 1 << m.from;
            self.player_bb[self.side_to_move as usize] |= 1 << m.from;

            self.board[m.to as usize] = m.capture_piece;
            self.board[m.from as usize] = m.piece;

            // 二歩フラグのundo
            if m.piece.get_piece_type() == PieceType::Pawn && m.promotion {
                self.pawn_flags[self.side_to_move as usize] |= 1 << (m.to % 5);
            }

            // 相手の駒を取っていた場合には、持ち駒から減らす
            if m.capture_piece != Piece::NoPiece {
                self.hand[self.side_to_move as usize]
                    [m.capture_piece.get_piece_type().get_raw() as usize - 2] -= 1;

                // Bitboardのundo
                self.piece_bb[m.capture_piece as usize] |= 1 << m.to;
                self.player_bb[self.side_to_move.get_op_color() as usize] |= 1 << m.to;

                // 二歩フラグのundo
                if m.capture_piece.get_piece_type() == PieceType::Pawn {
                    self.pawn_flags[self.side_to_move.get_op_color() as usize] |= 1 << (m.to % 5);
                }
            }
        }
    }

    /// 千日手かどうかを返す
    /// (千日手かどうか, 連続王手の千日手かどうか)
    pub fn is_repetition(&self) -> (bool, bool) {
        if self.ply == 0 {
            return (false, false);
        }

        let mut count = 0;

        let mut ply = self.ply as i32 - 2;
        while ply >= 0 {
            if self.hash[ply as usize] == self.hash[self.ply as usize] {
                count += 1;
            }

            // 現在の局面の1手前から数え始めているので、3回(+現在の局面 1回)で千日手
            if count == 3 {
                // 連続王手
                if self.sequent_check_count[self.ply as usize]
                    [self.side_to_move.get_op_color() as usize]
                    >= 7
                {
                    return (true, true);
                }

                return (true, false);
            }

            ply -= 2; // 繰り返し回数は、同じ手番の過去局面だけを見れば良い
        }

        return (false, false);
    }

    /// 現在の局面がこれまでに何回出てきたかを返す
    pub fn get_repetition(&self) -> usize {
        let mut count: usize = 0;

        let mut ply = self.ply as i32 - 2;
        while ply >= 0 {
            if self.hash[ply as usize] == self.hash[self.ply as usize] {
                count += 1;
            }

            ply -= 2; // 繰り返し回数は、同じ手番の過去局面だけを見れば良い
        }

        return count;
    }

    pub fn to_svg(&self) -> String {
        // ToDo:
        //   color_last_move: bool
        //   color_promoted_piece: bool
        //   p1_name: String
        //   p2_name: String

        let mut svg_text: String = String::new();

        svg_text.push_str("<svg width=\"448px\" height=\"384px\"\n     xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n");

        svg_text.push_str("  <rect x=\"64\" y=\"32\" width=\"320\" height=\"320\" fill=\"white\" stroke=\"black\" stroke-width=\"3\" />\n");

        for y in 0..5 {
            for x in 0..5 {
                svg_text.push_str(&format!("  <rect x=\"{}\" y=\"{}\" width=\"64\" height=\"64\" fill=\"white\" stroke=\"black\" stroke-width=\"1\" />\n",
                                    64 + 64 * x, 32 + 64 * y));
            }
        }

        for i in 0..SQUARE_NB {
            if self.board[i] != Piece::NoPiece {
                let kanji = piece_type_to_kanji(self.board[i].get_piece_type());

                let y = i / 5;
                let x = i % 5;

                if self.board[i].get_color() == Color::White {
                    svg_text.push_str(&format!("  <text x=\"{}\" y=\"{}\" font-family=\"serif\" font-size=\"42\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
                            96 + 64 * x, 64 + 64 * y, kanji));
                } else {
                    svg_text.push_str(&format!("  <text x=\"{}\" y=\"{}\" font-family=\"serif\" font-size=\"42\" text-anchor=\"middle\" dominant-baseline=\"central\" transform=\"rotate(180, {}, {})\">{}</text>\n",
                            96 + 64 * x, 64 + 64 * y, 96 + 64 * x, 64 + 64 * y, kanji));
                }
            }
        }

        {
            svg_text.push_str(&format!("  <text x=\"{}\" y=\"{}\" font-family=\"serif\" font-size=\"36\" writing-mode=\"tb\">&#9751;</text>\n", 420, 32));
            let mut hand_string = String::new();
            for piece_type in &HAND_PIECE_TYPE_ALL {
                if self.hand[Color::White as usize][*piece_type as usize - 2] != 0 {
                    hand_string.push_str(&piece_type_to_kanji(*piece_type));
                    if self.hand[Color::White as usize][*piece_type as usize - 2] == 2 {
                        hand_string.push_str(&"二".to_string());
                    }
                }
            }

            if !hand_string.is_empty() {
                svg_text.push_str(&format!("  <text x=\"{}\" y=\"{}\" font-family=\"serif\" font-size=\"36\" writing-mode=\"tb\" letter-spacing=\"1\">{}</text>\n", 420, 74, hand_string));
            }
        }

        {
            svg_text.push_str(&format!("  <text x=\"{}\" y=\"{}\" font-family=\"serif\" font-size=\"36\" writing-mode=\"tb\" transform=\"rotate(180, {}, {})\">&#9750;</text>\n", 32, 352, 32, 352));
            let mut hand_string = String::new();
            for piece_type in &HAND_PIECE_TYPE_ALL {
                if self.hand[Color::Black as usize][*piece_type as usize - 2] != 0 {
                    hand_string.push_str(&piece_type_to_kanji(*piece_type));
                    if self.hand[Color::Black as usize][*piece_type as usize - 2] == 2 {
                        hand_string.push_str(&"二".to_string());
                    }
                }
            }

            if !hand_string.is_empty() {
                svg_text.push_str(&format!("  <text x=\"{}\" y=\"{}\" font-family=\"serif\" font-size=\"36\" writing-mode=\"tb\" letter-spacing=\"1\" transform=\"rotate(180, {}, {})\">{}</text>\n", 32, 310, 32, 310, hand_string));
            }
        }

        svg_text.push_str("</svg>\n");

        return svg_text;
    }
}

#[test]
fn to_svg_test() {
    let mut position = Position::empty_board();
    position.set_start_position();
    println!("{}", position.to_svg());
}

impl Position {
    pub fn empty_board() -> Position {
        Position {
            side_to_move: Color::NoColor,
            board: [Piece::NoPiece; SQUARE_NB],
            hand: [[0; 5]; 2],
            pawn_flags: [0; 2],
            piece_bb: [0; Piece::BPawnX as usize + 1],
            player_bb: [0; 2],
            ply: 0,
            kif: [NULL_MOVE; MAX_PLY + 1],
            hash: [0; MAX_PLY + 1],
            adjacent_check_bb: [0; MAX_PLY + 1],
            long_check_bb: [0; MAX_PLY + 1],
            sequent_check_count: [[0; 2]; MAX_PLY + 1],
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

    fn set_check_bb(&mut self) {
        self.adjacent_check_bb[self.ply as usize] = 0;
        self.long_check_bb[self.ply as usize] = 0;

        let king_square =
            get_square(self.piece_bb[PieceType::King.get_piece(self.side_to_move) as usize]);

        assert!(king_square < SQUARE_NB);

        for piece_type in PIECE_TYPE_ALL.iter() {
            let check_bb = adjacent_attack(king_square, piece_type.get_piece(self.side_to_move))
                & self.piece_bb[piece_type.get_piece(self.side_to_move.get_op_color()) as usize];

            if check_bb != 0 {
                self.adjacent_check_bb[self.ply as usize] |= check_bb;
            }
        }

        let player_bb =
            self.player_bb[Color::White as usize] | self.player_bb[Color::Black as usize];

        // 角による王手
        let bishop_check_bb = bishop_attack(king_square, player_bb);
        self.long_check_bb[self.ply as usize] |= bishop_check_bb
            & self.piece_bb[PieceType::Bishop.get_piece(self.side_to_move.get_op_color()) as usize];
        self.long_check_bb[self.ply as usize] |= bishop_check_bb
            & self.piece_bb
                [PieceType::BishopX.get_piece(self.side_to_move.get_op_color()) as usize];

        // 飛車による王手
        let rook_check_bb = rook_attack(king_square, player_bb);
        self.long_check_bb[self.ply as usize] |= rook_check_bb
            & self.piece_bb[PieceType::Rook.get_piece(self.side_to_move.get_op_color()) as usize];
        self.long_check_bb[self.ply as usize] |= rook_check_bb
            & self.piece_bb[PieceType::RookX.get_piece(self.side_to_move.get_op_color()) as usize];
    }

    fn calculate_hash(&self) -> u64 {
        let mut hash: u64 = 0;

        for i in 0..SQUARE_NB {
            if self.board[i] != Piece::NoPiece {
                hash ^= ::zobrist::BOARD_TABLE[i][self.board[i] as usize];
            }
        }

        if self.side_to_move == Color::Black {
            hash |= 1;
        }

        return hash;
    }

    fn get_hash(&self) -> u64 {
        return self.hash[self.ply as usize];
    }

    pub fn get_adjacent_check_bb(&self) -> Bitboard {
        return self.adjacent_check_bb[self.ply as usize];
    }

    pub fn get_long_check_bb(&self) -> Bitboard {
        return self.long_check_bb[self.ply as usize];
    }

    pub fn get_check_bb(&self) -> Bitboard {
        return self.get_adjacent_check_bb() | self.get_long_check_bb();
    }

    pub fn get_sfen_position(&self) -> String {
        let mut sfen_position = String::new();

        let mut empty: u8 = 0;

        for i in 0..SQUARE_NB {
            if self.board[i] == Piece::NoPiece {
                empty += 1;
            } else {
                if empty > 0 {
                    sfen_position.push_str(&empty.to_string());
                }
                empty = 0;

                sfen_position.push_str(&piece_to_string(self.board[i]));
            }

            if i % 5 == 4 {
                if empty > 0 {
                    sfen_position.push_str(&empty.to_string());
                }
                empty = 0;

                if i != SQUARE_NB - 1 {
                    sfen_position.push('/');
                }
            }
        }

        sfen_position.push(' ');

        if self.side_to_move == Color::White {
            sfen_position.push('b');
        } else {
            sfen_position.push('w');
        }

        sfen_position.push(' ');

        let mut capture_flag = false;

        for piece_type in &HAND_PIECE_TYPE_ALL {
            if self.hand[Color::White as usize][*piece_type as usize - 2] > 0 {
                sfen_position.push_str(
                    &self.hand[Color::White as usize][*piece_type as usize - 2].to_string(),
                );
                sfen_position.push_str(&piece_to_string(piece_type.get_piece(Color::White)));
                capture_flag = true;
            }
            if self.hand[Color::Black as usize][*piece_type as usize - 2] > 0 {
                sfen_position.push_str(
                    &self.hand[Color::Black as usize][*piece_type as usize - 2].to_string(),
                );
                sfen_position.push_str(&piece_to_string(piece_type.get_piece(Color::Black)));
                capture_flag = true;
            }
        }

        if !capture_flag {
            sfen_position.push('-');
        }

        sfen_position.push(' ');
        sfen_position.push('1');

        return sfen_position;
    }

    pub fn generate_moves_with_option(
        &self,
        is_board: bool,
        is_hand: bool,
        allow_illegal: bool,
    ) -> std::vec::Vec<Move> {
        let mut moves: Vec<Move> = Vec::new();

        if is_board {
            let mut player_bb: Bitboard = self.player_bb[self.side_to_move as usize];

            while player_bb != 0 {
                let i = get_square(player_bb);
                player_bb ^= 1 << i;

                // 両王手がかかっているときは，玉を逃げる以外は非合法手
                if !allow_illegal
                    && get_counts(
                        self.adjacent_check_bb[self.ply as usize]
                            | self.long_check_bb[self.ply as usize],
                    ) > 1
                {
                    if self.board[i].get_piece_type() != PieceType::King {
                        continue;
                    }
                }

                // 飛び駒以外の駒の移動
                {
                    let mut move_tos: Bitboard = adjacent_attack(i, self.board[i]); // 利きの取得
                    move_tos = move_tos & !self.player_bb[self.side_to_move as usize]; // 自分の駒がある場所には動けない

                    while move_tos != 0 {
                        let move_to: usize = get_square(move_tos); // 行先を1か所取得する

                        // 近接王手がかかっていて，玉以外を動かす場合には，王手している駒を取るしかない
                        if !allow_illegal
                            && self.adjacent_check_bb[self.ply as usize] != 0
                            && self.board[i].get_piece_type() != PieceType::King
                            && (self.adjacent_check_bb[self.ply as usize] & (1 << move_to)) == 0
                        {
                            move_tos ^= 1 << move_to;
                            continue;
                        }

                        let capture_piece = self.board[move_to];
                        let (move_dir, _) = get_relation(i, move_to);

                        if (self.board[i] == Piece::WPawn && move_to < 5)
                            || (self.board[i] == Piece::BPawn && move_to >= 20)
                        {
                            // 行き場のない歩の不成の手は生成しない
                        } else {
                            moves.push(Move::board_move(
                                self.board[i],
                                i,
                                move_dir,
                                1,
                                move_to,
                                false,
                                capture_piece,
                            ));
                        }

                        // 成る手の生成
                        if self.board[i].is_raw()
                            && self.board[i].is_promotable()
                            && ((self.side_to_move == Color::White && (move_to < 5 || i < 5))
                                || (self.side_to_move == Color::Black
                                    && (move_to >= 20 || i >= 20)))
                        {
                            moves.push(Move::board_move(
                                self.board[i],
                                i,
                                move_dir,
                                1,
                                move_to,
                                true,
                                capture_piece,
                            ));
                        }

                        move_tos ^= 1 << move_to;
                    }
                }

                let all_player_bb =
                    self.player_bb[Color::White as usize] | self.player_bb[Color::Black as usize];

                // 飛び駒の移動
                // 角、馬
                if self.board[i].get_piece_type() == PieceType::Bishop
                    || self.board[i].get_piece_type() == PieceType::BishopX
                {
                    let mut move_tos: Bitboard = bishop_attack(i, all_player_bb);
                    move_tos &= !self.player_bb[self.side_to_move as usize];

                    while move_tos != 0 {
                        let move_to: usize = get_square(move_tos);

                        if !allow_illegal
                            && self.adjacent_check_bb[self.ply as usize] != 0
                            && self.board[i].get_piece_type() != PieceType::King
                            && (self.adjacent_check_bb[self.ply as usize] & (1 << move_to)) == 0
                        {
                            move_tos ^= 1 << move_to;
                            continue;
                        }

                        let capture_piece = self.board[move_to];
                        let (move_dir, amount) = get_relation(i, move_to);

                        moves.push(Move::board_move(
                            self.board[i],
                            i,
                            move_dir,
                            amount,
                            move_to,
                            false,
                            capture_piece,
                        ));

                        // 成る手の生成
                        if self.board[i].is_raw()
                            && self.board[i].is_promotable()
                            && ((self.side_to_move == Color::White && (move_to < 5 || i < 5))
                                || (self.side_to_move == Color::Black
                                    && (move_to >= 20 || i >= 20)))
                        {
                            moves.push(Move::board_move(
                                self.board[i],
                                i,
                                move_dir,
                                amount,
                                move_to,
                                true,
                                capture_piece,
                            ));
                        }

                        move_tos ^= 1 << move_to;
                    }
                }
                // 飛、龍
                else if self.board[i].get_piece_type() == PieceType::Rook
                    || self.board[i].get_piece_type() == PieceType::RookX
                {
                    let mut move_tos: Bitboard = rook_attack(i, all_player_bb);
                    move_tos &= !self.player_bb[self.side_to_move as usize];

                    while move_tos != 0 {
                        let move_to: usize = get_square(move_tos);

                        if !allow_illegal
                            && self.adjacent_check_bb[self.ply as usize] != 0
                            && self.board[i].get_piece_type() != PieceType::King
                            && (self.adjacent_check_bb[self.ply as usize] & (1 << move_to)) == 0
                        {
                            move_tos ^= 1 << move_to;
                            continue;
                        }

                        let capture_piece = self.board[move_to];
                        let (move_dir, amount) = get_relation(i, move_to);

                        moves.push(Move::board_move(
                            self.board[i],
                            i,
                            move_dir,
                            amount,
                            move_to,
                            false,
                            capture_piece,
                        ));

                        // 成る手の生成
                        if self.board[i].is_raw()
                            && self.board[i].is_promotable()
                            && ((self.side_to_move == Color::White && (move_to < 5 || i < 5))
                                || (self.side_to_move == Color::Black
                                    && (move_to >= 20 || i >= 20)))
                        {
                            moves.push(Move::board_move(
                                self.board[i],
                                i,
                                move_dir,
                                amount,
                                move_to,
                                true,
                                capture_piece,
                            ));
                        }

                        move_tos ^= 1 << move_to;
                    }
                }
            }
        }

        // 近接駒に王手されている場合、持ち駒を打つ手は全て非合法手
        if is_hand && (allow_illegal || self.adjacent_check_bb[self.ply as usize] == 0) {
            // 駒のない升を列挙
            let mut empty_squares: Vec<usize> = Vec::new();
            for i in 0..SQUARE_NB {
                if self.board[i] == Piece::NoPiece {
                    empty_squares.push(i);
                }
            }

            for piece_type in HAND_PIECE_TYPE_ALL.iter() {
                if self.hand[self.side_to_move as usize][*piece_type as usize - 2] > 0 {
                    for target in &empty_squares {
                        // 二歩は禁じ手
                        if *piece_type == PieceType::Pawn
                            && self.pawn_flags[self.side_to_move as usize] & (1 << (target % 5))
                                != 0
                        {
                            continue;
                        }

                        // 行き場のない駒を打たない
                        if *piece_type == PieceType::Pawn
                            && ((self.side_to_move == Color::White && *target < 5)
                                || (self.side_to_move == Color::Black && *target >= 20))
                        {
                            continue;
                        }

                        moves.push(Move::hand_move(
                            piece_type.get_piece(self.side_to_move),
                            *target,
                        ));
                    }
                }
            }
        }

        // 非合法手を取り除く
        if !allow_illegal {
            let king_square =
                get_square(self.piece_bb[PieceType::King.get_piece(self.side_to_move) as usize]);

            let mut index: usize = 0;

            loop {
                if index == moves.len() {
                    break;
                }

                let is_legal = |m: Move| -> bool {
                    if m.amount == 0 {
                        // 持ち駒を打つ場合
                        let player_bb: Bitboard = self.player_bb[Color::White as usize]
                            | self.player_bb[Color::Black as usize]
                            | (1 << m.to);

                        // 角による王手
                        let bishop_check_bb = bishop_attack(king_square, player_bb);
                        if bishop_check_bb
                            & self.piece_bb[PieceType::Bishop
                                .get_piece(self.side_to_move.get_op_color())
                                as usize]
                            != 0
                            || bishop_check_bb
                                & self.piece_bb[PieceType::BishopX
                                    .get_piece(self.side_to_move.get_op_color())
                                    as usize]
                                != 0
                        {
                            return false;
                        }

                        // 飛車による王手
                        let rook_check_bb = rook_attack(king_square, player_bb);
                        if rook_check_bb
                            & self.piece_bb[PieceType::Rook
                                .get_piece(self.side_to_move.get_op_color())
                                as usize]
                            != 0
                            || rook_check_bb
                                & self.piece_bb[PieceType::RookX
                                    .get_piece(self.side_to_move.get_op_color())
                                    as usize]
                                != 0
                        {
                            return false;
                        }
                    } else {
                        // 盤上の駒を動かす場合
                        if m.piece.get_piece_type() == PieceType::King {
                            // 王を動かす場合
                            let player_bb: Bitboard = (self.player_bb[Color::White as usize]
                                | self.player_bb[Color::Black as usize]
                                | (1 << m.to))
                                ^ (1 << m.from);

                            // 角による王手
                            let bishop_check_bb = bishop_attack(m.to as usize, player_bb);

                            if bishop_check_bb
                                & self.piece_bb[PieceType::Bishop
                                    .get_piece(self.side_to_move.get_op_color())
                                    as usize]
                                != 0
                                || bishop_check_bb
                                    & self.piece_bb[PieceType::BishopX
                                        .get_piece(self.side_to_move.get_op_color())
                                        as usize]
                                    != 0
                            {
                                return false;
                            }

                            // 飛車による王手
                            let rook_check_bb = rook_attack(m.to as usize, player_bb);

                            if rook_check_bb
                                & self.piece_bb[PieceType::Rook
                                    .get_piece(self.side_to_move.get_op_color())
                                    as usize]
                                != 0
                                || rook_check_bb
                                    & self.piece_bb[PieceType::RookX
                                        .get_piece(self.side_to_move.get_op_color())
                                        as usize]
                                    != 0
                            {
                                return false;
                            }

                            // 近接王手
                            for piece_type in PIECE_TYPE_ALL.iter() {
                                let check_bb = adjacent_attack(
                                    m.to as usize,
                                    piece_type.get_piece(self.side_to_move),
                                ) & self.piece_bb[piece_type
                                    .get_piece(self.side_to_move.get_op_color())
                                    as usize];

                                if check_bb != 0 {
                                    return false;
                                }
                            }
                        } else {
                            // 王以外を動かす場合
                            if get_counts(self.adjacent_check_bb[self.ply as usize]) > 1 {
                                // 近接駒に両王手されている場合は玉を動かさないといけない
                                return false;
                            } else if get_counts(self.adjacent_check_bb[self.ply as usize]) == 1 {
                                // 王手している近接駒を取る手でないといけない
                                if self.adjacent_check_bb[self.ply as usize] & (1 << m.to) == 0 {
                                    return false;
                                }
                            }

                            let player_bb: Bitboard = (self.player_bb[Color::White as usize]
                                | self.player_bb[Color::Black as usize]
                                | (1 << m.to))
                                ^ (1 << m.from);

                            // 角による王手
                            let bishop_check_bb =
                                bishop_attack(king_square, player_bb) & !(1 << m.to);
                            if bishop_check_bb
                                & self.piece_bb[PieceType::Bishop
                                    .get_piece(self.side_to_move.get_op_color())
                                    as usize]
                                != 0
                                || bishop_check_bb
                                    & self.piece_bb[PieceType::BishopX
                                        .get_piece(self.side_to_move.get_op_color())
                                        as usize]
                                    != 0
                            {
                                return false;
                            }

                            // 飛車による王手
                            let rook_check_bb = rook_attack(king_square, player_bb) & !(1 << m.to);

                            if rook_check_bb
                                & self.piece_bb[PieceType::Rook
                                    .get_piece(self.side_to_move.get_op_color())
                                    as usize]
                                != 0
                                || rook_check_bb
                                    & self.piece_bb[PieceType::RookX
                                        .get_piece(self.side_to_move.get_op_color())
                                        as usize]
                                    != 0
                            {
                                return false;
                            }
                        }
                    }

                    return true;
                }(moves[index]);

                if !is_legal {
                    moves.swap_remove(index);

                    continue;
                }

                index += 1;
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

        _ => Piece::NoPiece,
    }
}

fn piece_to_string(piece: Piece) -> String {
    match piece {
        Piece::WKing => "K".to_string(),
        Piece::WGold => "G".to_string(),
        Piece::WSilver => "S".to_string(),
        Piece::WBishop => "B".to_string(),
        Piece::WRook => "R".to_string(),
        Piece::WPawn => "P".to_string(),
        Piece::WSilverX => "+S".to_string(),
        Piece::WBishopX => "+B".to_string(),
        Piece::WRookX => "+R".to_string(),
        Piece::WPawnX => "+P".to_string(),

        Piece::BKing => "k".to_string(),
        Piece::BGold => "g".to_string(),
        Piece::BSilver => "s".to_string(),
        Piece::BBishop => "b".to_string(),
        Piece::BRook => "r".to_string(),
        Piece::BPawn => "p".to_string(),
        Piece::BSilverX => "+s".to_string(),
        Piece::BBishopX => "+b".to_string(),
        Piece::BRookX => "+r".to_string(),
        Piece::BPawnX => "+p".to_string(),

        _ => "ERROR".to_string(),
    }
}

fn piece_type_to_kanji(piece_type: PieceType) -> String {
    match piece_type {
        PieceType::King => "玉".to_string(),
        PieceType::Gold => "金".to_string(),
        PieceType::Silver => "銀".to_string(),
        PieceType::Bishop => "角".to_string(),
        PieceType::Rook => "飛".to_string(),
        PieceType::Pawn => "歩".to_string(),
        PieceType::SilverX => "全".to_string(),
        PieceType::BishopX => "馬".to_string(),
        PieceType::RookX => "龍".to_string(),
        PieceType::PawnX => "と".to_string(),

        _ => "".to_string(),
    }
}

#[test]
fn pawn_flags_test() {
    ::bitboard::init();

    const LOOP_NUM: i32 = 100000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            let mut pawn_flag: [[bool; 5]; 2] = [[false; 5]; 2];

            // 二歩フラグの差分更新が正しく動作していることを確認する
            for i in 0..SQUARE_NB {
                if position.board[i] == Piece::WPawn {
                    pawn_flag[Color::White as usize][(i % 5) as usize] = true;
                } else if position.board[i] == Piece::BPawn {
                    pawn_flag[Color::Black as usize][(i % 5) as usize] = true;
                }
            }
            for i in 0..5 {
                assert_eq!(
                    pawn_flag[Color::White as usize][i],
                    (position.pawn_flags[Color::White as usize] & (1 << i)) != 0
                );
                assert_eq!(
                    pawn_flag[Color::Black as usize][i],
                    (position.pawn_flags[Color::Black as usize] & (1 << i)) != 0
                );
            }

            let moves = position.generate_moves();
            if moves.len() == 0 {
                break;
            }

            // ランダムに局面を進める
            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

#[test]
fn move_do_undo_test() {
    ::bitboard::init();

    const LOOP_NUM: i32 = 10000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            let moves = position.generate_moves();

            for m in &moves {
                let mut temp_position = position;

                if m.capture_piece.get_piece_type() == PieceType::King {
                    continue;
                }

                temp_position.do_move(m);
                temp_position.undo_move();

                // do_move -> undo_moveで元の局面と一致するはず
                assert_eq!(position.side_to_move, temp_position.side_to_move);
                for i in 0..SQUARE_NB {
                    assert_eq!(position.board[i], temp_position.board[i]);
                }
                for i in 0..2 {
                    for j in 0..5 {
                        assert_eq!(position.hand[i][j], temp_position.hand[i][j]);
                    }
                }

                for i in 0..Piece::BPawnX as usize + 1 {
                    assert_eq!(position.piece_bb[i], temp_position.piece_bb[i]);
                }
                for i in 0..2 {
                    assert_eq!(position.player_bb[i], temp_position.player_bb[i]);
                }

                for i in 0..2 {
                    assert_eq!(position.pawn_flags[i], temp_position.pawn_flags[i]);
                }

                assert_eq!(position.ply, temp_position.ply);

                for i in 0..position.ply as usize {
                    assert!(position.kif[i] == temp_position.kif[i]);
                }

                assert_eq!(position.get_hash(), temp_position.get_hash());

                for i in 0..position.ply as usize {
                    assert_eq!(position.adjacent_check_bb[i], temp_position.adjacent_check_bb[i]);
                    assert_eq!(position.long_check_bb[i], temp_position.long_check_bb[i]);
                }

                for i in 0..position.ply as usize {
                    for j in 0..2 {
                        assert_eq!(
                            position.sequent_check_count[i][j],
                            temp_position.sequent_check_count[i][j]
                        );
                    }
                }
            }

            if moves.len() == 0 {
                break;
            }

            // ランダムに局面を進める
            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

#[test]
fn sfen_test() {
    ::bitboard::init();

    const LOOP_NUM: i32 = 1000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            let moves = position.generate_moves();

            {
                let mut temp_position = Position::empty_board();
                temp_position.set_sfen(&position.sfen(true));

                assert_eq!(position.side_to_move, temp_position.side_to_move);
                for i in 0..SQUARE_NB {
                    assert_eq!(position.board[i], temp_position.board[i]);
                }
                for i in 0..2 {
                    for j in 0..5 {
                        assert_eq!(position.hand[i][j], temp_position.hand[i][j]);
                    }
                }

                for i in 0..Piece::BPawnX as usize + 1 {
                    assert_eq!(position.piece_bb[i], temp_position.piece_bb[i]);
                }
                for i in 0..2 {
                    assert_eq!(position.player_bb[i], temp_position.player_bb[i]);
                }

                for i in 0..2 {
                    assert_eq!(position.pawn_flags[i], temp_position.pawn_flags[i]);
                }

                assert_eq!(position.ply, temp_position.ply);

                for i in 0..position.ply as usize {
                    assert!(position.kif[i] == temp_position.kif[i]);
                }

                assert_eq!(position.get_hash(), temp_position.get_hash());

                for i in 0..position.ply as usize {
                    assert_eq!(position.adjacent_check_bb[i], temp_position.adjacent_check_bb[i]);
                    assert_eq!(position.long_check_bb[i], temp_position.long_check_bb[i]);
                }

                for i in 0..position.ply as usize {
                    for j in 0..2 {
                        assert_eq!(
                            position.sequent_check_count[i][j],
                            temp_position.sequent_check_count[i][j]
                        );
                    }
                }
            }

            {
                let mut temp_position = Position::empty_board();
                temp_position.set_sfen(&position.sfen(false));

                assert_eq!(position.side_to_move, temp_position.side_to_move);
                for i in 0..SQUARE_NB {
                    assert_eq!(position.board[i], temp_position.board[i]);
                }
                for i in 0..2 {
                    for j in 0..5 {
                        assert_eq!(position.hand[i][j], temp_position.hand[i][j]);
                    }
                }

                for i in 0..Piece::BPawnX as usize + 1 {
                    assert_eq!(position.piece_bb[i], temp_position.piece_bb[i]);
                }
                for i in 0..2 {
                    assert_eq!(position.player_bb[i], temp_position.player_bb[i]);
                }

                for i in 0..2 {
                    assert_eq!(position.pawn_flags[i], temp_position.pawn_flags[i]);
                }
            }

            if moves.len() == 0 {
                break;
            }

            // ランダムに局面を進める
            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

#[test]
fn bitboard_test() {
    ::bitboard::init();

    const LOOP_NUM: i32 = 100000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            for i in 0..SQUARE_NB {
                if position.board[i] == Piece::NoPiece {
                    continue;
                }

                assert!(position.piece_bb[position.board[i] as usize] & (1 << i) != 0);
            }

            let moves = position.generate_moves();
            if moves.len() == 0 {
                break;
            }

            // ランダムに局面を進める
            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

#[test]
fn no_legal_move_test() {
    ::bitboard::init();

    static CHECKMATE_SFEN1: &str = "5/5/2p2/2g2/2K2 b P 1";
    static CHECKMATE_SFEN2: &str = "4k/1s1gp/p4/g1BS1/1KR2 b BRg 1";
    static CHECKMATE_SFEN3: &str = "4k/2G2/5/5/4R w - 1";
    static CHECKMATE_SFEN4: &str = "r4/5/5/2g2/K4 b - 1";
    static CHECKMATE_SFEN5: &str = "2G1k/5/4P/5/B4 w - 1";
    static CHECKMATE_SFEN6: &str = "4b/5/p4/5/K1g2 b - 1";
    static CHECKMATE_SFEN7: &str = "k1G2/5/P4/5/4B w - 1";
    static CHECKMATE_SFEN8: &str = "b4/5/4p/5/2g1K b - 1";
    static CHECKMATE_SFEN9: &str = "R4/2G1k/5/4P/1B3 w - 1";
    static CHECKMATE_SFEN10: &str = "r4/2g1K/5/4g/1b3 b - 1";

    let mut position = Position::empty_board();

    position.set_sfen(CHECKMATE_SFEN1);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN2);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN3);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN4);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN5);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN6);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN7);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN8);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN9);
    assert_eq!(position.generate_moves().len(), 0);

    position.set_sfen(CHECKMATE_SFEN10);
    assert_eq!(position.generate_moves().len(), 0);
}

#[test]
fn not_checkmate_positions() {
    ::bitboard::init();

    static NOT_CHECKMATE_SFEN1: &str = "rb1gk/1s2R/5/P1B2/KGS2 w P 1";

    let mut position = Position::empty_board();

    position.set_sfen(NOT_CHECKMATE_SFEN1);
    assert!(position.generate_moves().len() > 0);
}

#[test]
fn no_king_capture_move_in_legal_moves_test() {
    ::bitboard::init();

    const LOOP_NUM: i32 = 100000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            let moves = position.generate_moves();

            for m in &moves {
                // 玉が取られる手は生成しないはず
                // -> 玉が取れる局面に遭遇しないはず
                assert!(m.capture_piece.get_piece_type() != PieceType::King);
            }

            // ランダムに局面を進める
            if moves.len() == 0 {
                break;
            }

            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

#[test]
fn generate_moves_test() {
    ::bitboard::init();

    const LOOP_NUM: i32 = 10000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            let moves = position.generate_moves();
            let allow_illegal_moves = position.generate_moves_with_option(true, true, true);

            let mut legal_move_count = allow_illegal_moves.len();
            for m in allow_illegal_moves {
                position.do_move(&m);

                let all_moves = position.generate_moves_with_option(true, true, true);

                for m2 in all_moves {
                    if m2.capture_piece.get_piece_type() == PieceType::King {
                        legal_move_count -= 1;
                        break;
                    }
                }

                position.undo_move();
            }

            assert_eq!(moves.len(), legal_move_count);

            // ランダムに局面を進める
            if moves.len() == 0 {
                break;
            }
            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

#[test]
fn hash_test() {
    ::bitboard::init();
    ::zobrist::init();

    const LOOP_NUM: i32 = 100000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            let moves = position.generate_moves();

            if moves.len() == 0 {
                break;
            }

            // 差分計算と全計算の値が一致することを確認する
            assert_eq!(position.get_hash(), position.calculate_hash());

            // 手番bitと手番が一致することを確認する
            assert_eq!(position.side_to_move == Color::Black, position.get_hash() & 1 != 0);

            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

#[test]
fn is_repetition_test() {
    ::bitboard::init();
    ::zobrist::init();

    let mut position = Position::empty_board();

    static START_POSITION_SFEN: &str = "rbsgk/4p/5/P4/KGSBR b - 1";
    static REPETITION_SFEN: &str = "rbsgk/4p/5/P4/KGSBR b - 1 moves 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a";
    static CHECK_REPETITION_SFEN: &str = "2k2/5/5/5/2K2 b R 1 moves R*3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c";
    static NOT_REPETITION_SFEN: &str =
        "rbsgk/4p/5/P4/KGSBR b - 1 moves 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a";
    static NOT_CHECK_REPETITION_SFEN: &str =
        "2k2/5/5/5/2K2 b R 1 moves R*3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a";

    position.set_sfen(START_POSITION_SFEN);
    assert_eq!(position.is_repetition(), (false, false));

    position.set_sfen(REPETITION_SFEN);
    assert_eq!(position.is_repetition(), (true, false));

    position.set_sfen(CHECK_REPETITION_SFEN);
    assert_eq!(position.is_repetition(), (true, true));

    position.set_sfen(NOT_REPETITION_SFEN);
    assert_eq!(position.is_repetition(), (false, false));

    position.set_sfen(NOT_CHECK_REPETITION_SFEN);
    assert_eq!(position.is_repetition(), (false, false));
}

#[test]
fn get_repetition_test() {
    ::bitboard::init();
    ::zobrist::init();

    let mut position = Position::empty_board();

    static START_POSITION_SFEN: &str = "rbsgk/4p/5/P4/KGSBR b - 1";
    static REPETITION_SFEN: &str = "rbsgk/4p/5/P4/KGSBR b - 1 moves 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a";
    static CHECK_REPETITION_SFEN: &str = "2k2/5/5/5/2K2 b R 1 moves R*3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c";
    static NOT_REPETITION_SFEN: &str =
        "rbsgk/4p/5/P4/KGSBR b - 1 moves 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a";
    static NOT_CHECK_REPETITION_SFEN: &str =
        "2k2/5/5/5/2K2 b R 1 moves R*3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a";

    position.set_sfen(START_POSITION_SFEN);
    assert_eq!(position.get_repetition(), 0);

    position.set_sfen(REPETITION_SFEN);
    assert_eq!(position.get_repetition(), 3);

    position.set_sfen(CHECK_REPETITION_SFEN);
    assert_eq!(position.get_repetition(), 3);

    position.set_sfen(NOT_REPETITION_SFEN);
    assert_eq!(position.get_repetition(), 2);

    position.set_sfen(NOT_CHECK_REPETITION_SFEN);
    assert_eq!(position.get_repetition(), 2);
}

#[test]
fn sfen_to_move_test() {
    ::bitboard::init();
    ::zobrist::init();

    const LOOP_NUM: i32 = 10000;

    let mut position = Position::empty_board();

    let mut rng = rand::thread_rng();

    for _ in 0..LOOP_NUM {
        position.set_start_position();

        while position.ply < MAX_PLY as u16 {
            let moves = position.generate_moves();

            if moves.len() == 0 {
                break;
            }

            for m in &moves {
                let sfen_move = position.sfen_to_move(m.sfen());
                assert_eq!(sfen_move, *m);
            }

            let random_move = moves.choose(&mut rng).unwrap();
            position.do_move(random_move);
        }
    }
}

#[test]
fn init_position_moves_test() {
    ::bitboard::init();
    ::zobrist::init();

    let mut position = Position::empty_board();
    position.set_start_position();
    let moves = position.generate_moves();

    assert_eq!(moves.len(), 14);
}
