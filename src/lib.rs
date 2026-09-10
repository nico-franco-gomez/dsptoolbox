use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

#[pyfunction]
fn warp_time_series<'py>(
    py: Python<'py>,
    time_data: PyReadonlyArray2<'py, f64>,
    warping_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let time_data = time_data.as_array();
    let n_samples = time_data.shape()[0];
    let n_channels = time_data.shape()[1];
    let mut warped_time_data = Array2::zeros((n_samples, n_channels));

    if n_samples == 0 {
        return Err(PyIndexError::new_err(
            "index 0 is out of bounds for axis 0 with size 0",
        ));
    }

    let mut dirac = vec![0.0; n_samples];
    dirac[0] = 1.0;

    for channel in 0..n_channels {
        warped_time_data[(0, channel)] = time_data[(0, channel)];
    }

    for sample in 1..n_samples {
        let mut previous_input = dirac[0];
        let mut previous_output = -warping_factor * previous_input;
        dirac[0] = previous_output;

        for time in 1..n_samples {
            let input = dirac[time];
            let output =
                -warping_factor * input + previous_input + warping_factor * previous_output;
            dirac[time] = output;
            previous_input = input;
            previous_output = output;
        }

        for time in 0..n_samples {
            for channel in 0..n_channels {
                warped_time_data[(time, channel)] += dirac[time] * time_data[(sample, channel)];
            }
        }
    }

    Ok(PyArray2::from_owned_array(py, warped_time_data))
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(warp_time_series, module)?)?;
    Ok(())
}
