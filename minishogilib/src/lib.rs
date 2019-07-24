#[macro_use]
extern crate lazy_static;
extern crate bitintr;
extern crate numpy;
extern crate pyo3;
extern crate rand;

pub mod bitboard;
pub mod r#move;
pub mod neuralnetwork;
pub mod position;
pub mod types;
pub mod zobrist;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use position::*;
use r#move::*;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn minishogilib(_py: Python, m: &PyModule) -> PyResult<()> {
    r#move::init();
    bitboard::init();
    zobrist::init();

    m.add_wrapped(wrap_pyfunction!(version))?;

    m.add_class::<Position>()?;
    m.add_class::<Move>()?;

    Ok(())
}
