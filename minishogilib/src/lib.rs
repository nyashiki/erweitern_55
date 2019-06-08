extern crate pyo3;

pub mod types;
pub mod position;
pub mod r#move;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use position::*;

#[pymodule]
fn minishogilib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Position>()?;

    Ok(())
}
