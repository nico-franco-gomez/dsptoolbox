use numpy::{PyReadonlyArray1, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn add_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lattice_filtering_fir, module)?)?;
    Ok(())
}

#[pyfunction]
fn lattice_filtering_fir(
    k: PyReadonlyArray1<'_, f64>,
    mut td: PyReadwriteArray2<'_, f64>,
    mut state: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    let k = k.as_array();
    let mut td = td.as_array_mut();
    let mut state = state.as_array_mut();

    if state.shape()[0] != k.len() {
        return Err(PyValueError::new_err(
            "state length must match filter order",
        ));
    }
    if state.shape()[1] != td.shape()[1] {
        return Err(PyValueError::new_err(
            "state channels must match time-data channels",
        ));
    }

    for channel in 0..td.shape()[1] {
        for sample in 0..td.shape()[0] {
            let mut output = td[(sample, channel)];
            let mut previous = output;

            for coefficient in 0..k.len() {
                let next = -output * k[coefficient] + state[(coefficient, channel)];
                output -= state[(coefficient, channel)] * k[coefficient];
                state[(coefficient, channel)] = previous;
                previous = next;
            }

            td[(sample, channel)] = output;
        }
    }

    Ok(())
}
