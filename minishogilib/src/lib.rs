#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate bitintr;
extern crate pyo3;

pub mod types;
pub mod position;
pub mod r#move;
pub mod bitboard;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use position::*;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn minishogilib(_py: Python, m: &PyModule) -> PyResult<()> {
    bitboard::init();

    m.add_wrapped(wrap_pyfunction!(version))?;
    m.add_class::<Position>()?;

    Ok(())
}
