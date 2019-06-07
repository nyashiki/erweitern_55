use types::*;

pub struct Position {
    pub side_to_move: Color,
    pub board: [Piece; SQUARE_NB],
    pub hand: [[u8; 5]; 2],
    pub ply: u16
}

impl Position {
    pub fn new() -> Position {
        Position {
            side_to_move: Color::NoColor,
            board: [Piece::NoPiece; SQUARE_NB],
            hand: [[0; 5]; 2],
            ply: 0
        }
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
            print!("{}: {}, ", hand_str[i], self.hand[(Color::White as usize)][i]);
        }
        println!("");

        println!("ply: {}", self.ply);
    }
}
