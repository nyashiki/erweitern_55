#[macro_use]
extern crate lazy_static;
extern crate bitintr;
extern crate numpy;
extern crate pyo3;
extern crate rand;

pub mod bitboard;
pub mod mcts;
pub mod r#move;
pub mod neuralnetwork;
pub mod position;
pub mod types;
pub mod zobrist;

use pyo3::prelude::*;

use mcts::*;
use position::*;
use r#move::*;

#[pymodule]
fn minishogilib(_py: Python, m: &PyModule) -> PyResult<()> {
    r#move::init();
    bitboard::init();
    zobrist::init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<Position>()?;
    m.add_class::<MCTS>()?;
    m.add_class::<Move>()?;

    Ok(())
}
