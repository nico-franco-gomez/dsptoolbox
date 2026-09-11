use pyo3::prelude::*;

mod realtime;
mod transforms;

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    realtime::add_functions(module)?;
    transforms::add_functions(module)?;
    Ok(())
}
