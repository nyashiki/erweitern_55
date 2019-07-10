use types::*;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

lazy_static! {
    pub static ref BOARD_TABLE: [[u64; Piece::BPawnX as usize + 1]; SQUARE_NB] = {
        let mut table: [[u64; Piece::BPawnX as usize + 1]; SQUARE_NB] =
            [[0; Piece::BPawnX as usize + 1]; SQUARE_NB];

        let mut rng: StdRng = SeedableRng::from_seed([0; 32]);

        for i in 0..SQUARE_NB {
            for j in 0..Piece::BPawnX as usize + 1 {
                table[i][j] = rng.gen::<u64>() << 1;
            }
        }

        return table;
    };
}

pub fn init() {
    lazy_static::initialize(&BOARD_TABLE);
}
