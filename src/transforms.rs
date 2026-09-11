use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

pub fn add_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(morlet_wavelet, module)?)?;
    module.add_function(wrap_pyfunction!(laguerre, module)?)?;
    module.add_function(wrap_pyfunction!(warp_time_series, module)?)?;
    module.add_function(wrap_pyfunction!(squeeze_scalogram, module)?)?;
    Ok(())
}

#[pyfunction]
fn morlet_wavelet<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, Complex64>,
    inds: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let base = base.as_array();
    let inds = inds.as_array();

    if base.is_empty() {
        return Err(PyValueError::new_err("base wavelet must not be empty"));
    }

    let valid_indices: Vec<(usize, usize)> = inds
        .iter()
        .enumerate()
        .map(|(position, index)| (position, *index as usize))
        .filter(|(_, index)| *index < base.len())
        .collect();

    if valid_indices.is_empty() {
        return Err(PyValueError::new_err(
            "wavelet indices must contain a valid base-wavelet index",
        ));
    }

    let mut output = Array1::<Complex64>::zeros(valid_indices.len());
    for (output_index, (input_index, base_index)) in valid_indices.iter().enumerate() {
        if output_index + 1 == valid_indices.len() {
            output[output_index] = base[*base_index];
            continue;
        }

        if *base_index + 1 >= base.len() {
            continue;
        }

        let interpolation = inds[*input_index] - *base_index as f64;
        output[output_index] = base[*base_index]
            + (base[*base_index + 1] - base[*base_index]) * interpolation;
    }

    Ok(PyArray1::from_owned_array(py, output))
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

    let buffer_len = n_samples * n_channels;
    let mut filtered = vec![0.0; buffer_len];
    let normalization = (1.0 - warping_factor.powi(2)).sqrt();

    if n_channels == 1 {
        let mut previous_output = 0.0;
        if let Some(time_data_slice) = time_data.as_slice() {
            for sample in 0..n_samples {
                let input = time_data_slice[n_samples - sample - 1];
                previous_output = normalization * input - warping_factor * previous_output;
                filtered[sample] = previous_output;
            }
        } else {
            for sample in 0..n_samples {
                let input = time_data[(n_samples - sample - 1, 0)];
                previous_output = normalization * input - warping_factor * previous_output;
                filtered[sample] = previous_output;
            }
        }

        let mut output = vec![0.0; buffer_len];
        output[0] = filtered[n_samples - 1];
        for output_sample in output.iter_mut().skip(1) {
            let mut previous_input = filtered[0];
            let mut previous_output = warping_factor * previous_input;
            filtered[0] = previous_output;
            for filtered_sample in filtered.iter_mut().skip(1) {
                let input = *filtered_sample;
                let current_output =
                    warping_factor * input + previous_input - warping_factor * previous_output;
                *filtered_sample = current_output;
                previous_input = input;
                previous_output = current_output;
            }
            *output_sample = filtered[n_samples - 1];
        }

        let output = Array2::from_shape_vec((n_samples, n_channels), output)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        return Ok(PyArray2::from_owned_array(py, output));
    }

    let mut previous_outputs = vec![0.0; n_channels];
    if let Some(time_data_slice) = time_data.as_slice() {
        for sample in 0..n_samples {
            let input_offset = (n_samples - sample - 1) * n_channels;
            let filtered_offset = sample * n_channels;
            for channel in 0..n_channels {
                let current_output = normalization * time_data_slice[input_offset + channel]
                    - warping_factor * previous_outputs[channel];
                filtered[filtered_offset + channel] = current_output;
                previous_outputs[channel] = current_output;
            }
        }
    } else {
        for sample in 0..n_samples {
            let input_sample = n_samples - sample - 1;
            let filtered_offset = sample * n_channels;
            for channel in 0..n_channels {
                let current_output = normalization * time_data[(input_sample, channel)]
                    - warping_factor * previous_outputs[channel];
                filtered[filtered_offset + channel] = current_output;
                previous_outputs[channel] = current_output;
            }
        }
    }

    let mut output = vec![0.0; buffer_len];
    let last_offset = (n_samples - 1) * n_channels;
    output[..n_channels].copy_from_slice(&filtered[last_offset..last_offset + n_channels]);

    let mut previous_inputs = vec![0.0; n_channels];
    for stage in 1..n_samples {
        for channel in 0..n_channels {
            let previous_input = filtered[channel];
            let current_output = warping_factor * previous_input;
            previous_inputs[channel] = previous_input;
            previous_outputs[channel] = current_output;
            filtered[channel] = current_output;
        }

        for sample in 1..n_samples {
            let filtered_offset = sample * n_channels;
            for channel in 0..n_channels {
                let input = filtered[filtered_offset + channel];
                let current_output = warping_factor * input + previous_inputs[channel]
                    - warping_factor * previous_outputs[channel];
                filtered[filtered_offset + channel] = current_output;
                previous_inputs[channel] = input;
                previous_outputs[channel] = current_output;
            }
        }

        let output_offset = stage * n_channels;
        output[output_offset..output_offset + n_channels]
            .copy_from_slice(&filtered[last_offset..last_offset + n_channels]);
    }

    let output = Array2::from_shape_vec((n_samples, n_channels), output)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
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

        for dirac_sample in dirac.iter_mut().skip(1) {
            let input = *dirac_sample;
            let output =
                -warping_factor * input + previous_input + warping_factor * previous_output;
            *dirac_sample = output;
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
    freqs: PyReadonlyArray1<'py, f64>,
    fs: f64,
    delta_w: f64,
    apply_frequency_normalization: bool,
    gradient: PyReadonlyArray3<'py, Complex64>,
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
