use pyo3::prelude::*;
mod kmeans;
mod linreg;
mod utils;

/// Prints a message.
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from rs_algos!".into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rs_algos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<kmeans::KMeansRust>()?;
    m.add_class::<linreg::LinRegGDRust>()?;
    Ok(())
}
