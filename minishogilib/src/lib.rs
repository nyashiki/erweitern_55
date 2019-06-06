extern crate pyo3;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn hello_world() -> () {
  println!("Hello World in Rust!");
}

#[pymodule]
fn minishogilib(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_wrapped(wrap_pyfunction!(hello_world))?;

  Ok(())
}

