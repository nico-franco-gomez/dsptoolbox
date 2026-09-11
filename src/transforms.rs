use ndarray::{Array2, Array3};
use num_complex::Complex64;
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

pub fn add_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(laguerre, module)?)?;
    module.add_function(wrap_pyfunction!(warp_time_series, module)?)?;
    module.add_function(wrap_pyfunction!(squeeze_scalogram, module)?)?;
    Ok(())
}

#[pyfunction]
fn laguerre<'py>(
    py: Python<'py>,
    time_data: PyReadonlyArray2<'py, f64>,
    warping_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let time_data = time_data.as_array();
    let (n_samples, n_channels) = time_data.dim();
    if n_samples == 0 {
        return Err(PyIndexError::new_err(
            "index 0 is out of bounds for axis 0 with size 0",
        ));
    }

    let mut filtered = Array2::zeros((n_samples, n_channels));
    for sample in 0..n_samples {
        for channel in 0..n_channels {
            filtered[(sample, channel)] = time_data[(n_samples - sample - 1, channel)];
        }
    }

    let normalization = (1.0 - warping_factor.powi(2)).sqrt();
    for channel in 0..n_channels {
        let mut previous_output = normalization * filtered[(0, channel)];
        filtered[(0, channel)] = previous_output;
        for sample in 1..n_samples {
            previous_output =
                normalization * filtered[(sample, channel)] - warping_factor * previous_output;
            filtered[(sample, channel)] = previous_output;
        }
    }

    let mut output = Array2::zeros((n_samples, n_channels));
    for channel in 0..n_channels {
        output[(0, channel)] = filtered[(n_samples - 1, channel)];
    }

    for stage in 1..n_samples {
        for channel in 0..n_channels {
            let mut previous_input = filtered[(0, channel)];
            let mut previous_output = warping_factor * previous_input;
            filtered[(0, channel)] = previous_output;

            for sample in 1..n_samples {
                let input = filtered[(sample, channel)];
                let current_output =
                    warping_factor * input + previous_input - warping_factor * previous_output;
                filtered[(sample, channel)] = current_output;
                previous_input = input;
                previous_output = current_output;
            }
            output[(stage, channel)] = filtered[(n_samples - 1, channel)];
        }
    }

    Ok(PyArray2::from_owned_array(py, output))
}

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

#[pyfunction]
fn squeeze_scalogram<'py>(
    py: Python<'py>,
    scalogram: PyReadonlyArray3<'py, Complex64>,
    gradient: PyReadonlyArray3<'py, Complex64>,
    freqs: PyReadonlyArray1<'py, f64>,
    fs: f64,
    delta_w: f64,
    apply_frequency_normalization: bool,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    let scalogram = scalogram.as_array();
    let gradient = gradient.as_array();
    let freqs = freqs.as_array();
    let (n_frequencies, n_times, n_channels) = scalogram.dim();

    if gradient.dim() != scalogram.dim() {
        return Err(PyValueError::new_err(
            "gradient shape must match scalogram shape",
        ));
    }
    if freqs.len() != n_frequencies {
        return Err(PyValueError::new_err(
            "frequency vector length must match scalogram frequency dimension",
        ));
    }
    if n_times == 0 {
        return Err(PyValueError::new_err(
            "scalogram must contain at least one time sample",
        ));
    }

    let mut sync = Array3::<Complex64>::zeros((n_frequencies, n_times, n_channels));
    let normalizations: Vec<f64> = if apply_frequency_normalization {
        freqs
            .iter()
            .map(|frequency| (frequency / fs).powf(1.5))
            .collect()
    } else {
        Vec::new()
    };

    for channel in 0..n_channels {
        for time in 0..n_times {
            for source_frequency in 0..n_frequencies {
                let value = scalogram[(source_frequency, time, channel)];
                if value.norm_sqr() <= 1e-40 {
                    continue;
                }

                let phase_frequency = (gradient[(source_frequency, time, channel)] / value)
                    .im
                    .abs()
                    * fs
                    / (2.0 * std::f64::consts::PI);

                let mut target_frequency = 0;
                let mut minimum_difference = (freqs[0] - phase_frequency).abs();
                for target in 1..n_frequencies {
                    let difference = (freqs[target] - phase_frequency).abs();
                    if difference < minimum_difference {
                        minimum_difference = difference;
                        target_frequency = target;
                    }
                }

                if minimum_difference > delta_w * freqs[source_frequency] {
                    continue;
                }

                let contribution = if apply_frequency_normalization {
                    value * normalizations[source_frequency]
                } else {
                    value
                };
                sync[(target_frequency, time, channel)] += contribution;
            }
        }
    }

    Ok(PyArray3::from_owned_array(py, sync))
}
