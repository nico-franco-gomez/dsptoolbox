use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3,
    PyReadwriteArray4,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn add_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fir_filtering_sample, module)?)?;
    module.add_function(wrap_pyfunction!(iir_filtering_sample, module)?)?;
    module.add_function(wrap_pyfunction!(state_space_filtering_sample, module)?)?;
    module.add_function(wrap_pyfunction!(kautz_filtering_sample, module)?)?;
    module.add_function(wrap_pyfunction!(parallel_filtering_sample, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_filtering_fir, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_filtering_fir_sample, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_filtering_iir, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_filtering_iir_sample, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_filtering_sos, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_filtering_sos_sample, module)?)?;
    module.add_function(wrap_pyfunction!(warped_fir_filtering, module)?)?;
    module.add_function(wrap_pyfunction!(warped_fir_filtering_sample, module)?)?;
    module.add_function(wrap_pyfunction!(warped_fir_filtering_block, module)?)?;
    module.add_function(wrap_pyfunction!(warped_iir_filtering, module)?)?;
    module.add_function(wrap_pyfunction!(warped_iir_filtering_sample, module)?)?;
    module.add_function(wrap_pyfunction!(warped_iir_filtering_block, module)?)?;
    Ok(())
}

#[pyfunction]
fn state_space_filtering_sample(
    a: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    d: f64,
    input: f64,
    mut state: PyReadwriteArray2<'_, f64>,
    channel: usize,
) -> PyResult<f64> {
    let a = a.as_array();
    let b = b.as_array();
    let c = c.as_array();
    let mut state = state.as_array_mut();

    if a.shape()[0] != a.shape()[1]
        || b.len() != a.shape()[0]
        || c.len() != a.shape()[0]
        || state.shape()[0] != a.shape()[0]
        || state.shape()[1] <= channel
    {
        return Err(PyValueError::new_err(
            "state-space matrices or state dimensions are invalid",
        ));
    }

    let mut output = d * input;
    for state_index in 0..a.shape()[0] {
        output += c[state_index] * state[(state_index, channel)];
    }

    let mut next_state = vec![0.0; a.shape()[0]];
    for state_index in 0..a.shape()[0] {
        let mut value = b[state_index] * input;
        for previous_state in 0..a.shape()[1] {
            value += a[(state_index, previous_state)] * state[(previous_state, channel)];
        }
        next_state[state_index] = value;
    }
    for state_index in 0..a.shape()[0] {
        state[(state_index, channel)] = next_state[state_index];
    }

    Ok(output)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn kautz_filtering_sample(
    real_poles: PyReadonlyArray1<'_, f64>,
    real_coefficients: PyReadonlyArray1<'_, f64>,
    complex_q: PyReadonlyArray1<'_, f64>,
    complex_r: PyReadonlyArray1<'_, f64>,
    complex_coefficients: PyReadonlyArray1<'_, f64>,
    input: f64,
    mut real_state: PyReadwriteArray2<'_, f64>,
    mut real_advance_state: PyReadwriteArray2<'_, f64>,
    mut complex_state: PyReadwriteArray4<'_, f64>,
    mut complex_advance_state: PyReadwriteArray3<'_, f64>,
    channel: usize,
) -> PyResult<f64> {
    let real_poles = real_poles.as_array();
    let real_coefficients = real_coefficients.as_array();
    let complex_q = complex_q.as_array();
    let complex_r = complex_r.as_array();
    let complex_coefficients = complex_coefficients.as_array();
    let mut real_state = real_state.as_array_mut();
    let mut real_advance_state = real_advance_state.as_array_mut();
    let mut complex_state = complex_state.as_array_mut();
    let mut complex_advance_state = complex_advance_state.as_array_mut();

    if real_poles.len() != real_coefficients.len()
        || complex_q.len() != complex_r.len()
        || complex_coefficients.len() != 2 * complex_q.len()
        || real_state.dim() != (real_poles.len(), real_state.shape()[1])
        || real_advance_state.dim() != (real_poles.len(), real_advance_state.shape()[1])
        || real_state.shape()[1] <= channel
        || real_advance_state.shape()[1] <= channel
        || complex_state.shape() != [complex_q.len(), 2, 2, complex_state.shape()[3]]
        || complex_advance_state.shape() != [complex_q.len(), 2, complex_advance_state.shape()[2]]
        || complex_state.shape()[3] <= channel
        || complex_advance_state.shape()[2] <= channel
    {
        return Err(PyValueError::new_err(
            "invalid Kautz coefficients or state dimensions",
        ));
    }

    let mut input = input;
    let mut output = 0.0;

    for section in 0..real_poles.len() {
        let pole = real_poles[section];
        let filter_output = (1.0 - pole * pole).sqrt() * input + real_state[(section, channel)];
        real_state[(section, channel)] = pole * filter_output;
        output += filter_output * real_coefficients[section];

        let advanced_output = -pole * input + real_advance_state[(section, channel)];
        real_advance_state[(section, channel)] = input + pole * advanced_output;
        input = advanced_output;
    }

    for section in 0..complex_q.len() {
        let q = complex_q[section];
        let r = complex_r[section];
        let first_scale = ((1.0 - r) * (1.0 + r - q) / 2.0).sqrt();
        let second_scale = ((1.0 - r) * (1.0 + r + q) / 2.0).sqrt();

        let first_output = first_scale * input + complex_state[(section, 0, 0, channel)];
        complex_state[(section, 0, 0, channel)] =
            -first_scale * input - first_output * q + complex_state[(section, 0, 1, channel)];
        complex_state[(section, 0, 1, channel)] = -first_output * r;
        output += first_output * complex_coefficients[2 * section];

        let second_output = second_scale * input + complex_state[(section, 1, 0, channel)];
        complex_state[(section, 1, 0, channel)] =
            second_scale * input - second_output * q + complex_state[(section, 1, 1, channel)];
        complex_state[(section, 1, 1, channel)] = -second_output * r;
        output += second_output * complex_coefficients[2 * section + 1];

        let advanced_output = r * input + complex_advance_state[(section, 0, channel)];
        complex_advance_state[(section, 0, channel)] =
            q * input - advanced_output * q + complex_advance_state[(section, 1, channel)];
        complex_advance_state[(section, 1, channel)] = input - advanced_output * r;
        input = advanced_output;
    }

    Ok(output)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn parallel_filtering_sample(
    iir_b: PyReadonlyArray2<'_, f64>,
    iir_a: PyReadonlyArray2<'_, f64>,
    fir_b: PyReadonlyArray1<'_, f64>,
    delay_b: PyReadonlyArray1<'_, f64>,
    input: f64,
    mut iir_state: PyReadwriteArray3<'_, f64>,
    mut fir_state: PyReadwriteArray2<'_, f64>,
    mut fir_index: PyReadwriteArray1<'_, i64>,
    mut delay_state: PyReadwriteArray2<'_, f64>,
    mut delay_index: PyReadwriteArray1<'_, i64>,
    channel: usize,
) -> PyResult<f64> {
    let iir_b = iir_b.as_array();
    let iir_a = iir_a.as_array();
    let fir_b = fir_b.as_array();
    let delay_b = delay_b.as_array();
    let mut iir_state = iir_state.as_array_mut();
    let mut fir_state = fir_state.as_array_mut();
    let mut fir_index = fir_index.as_array_mut();
    let mut delay_state = delay_state.as_array_mut();
    let mut delay_index = delay_index.as_array_mut();

    if iir_b.shape() != iir_a.shape()
        || iir_b.shape()[1] != 3
        || iir_state.shape() != [iir_b.shape()[0], 2, iir_state.shape()[2]]
        || fir_state.shape()[0] != fir_b.len().saturating_sub(1)
        || delay_state.shape()[0] != delay_b.len().saturating_sub(1)
        || fir_state.shape()[1] <= channel
        || delay_state.shape()[1] <= channel
        || iir_state.shape()[2] <= channel
        || fir_index.len() <= channel
        || delay_index.len() <= channel
    {
        return Err(PyValueError::new_err(
            "invalid parallel filter coefficients or state dimensions",
        ));
    }

    let mut input = input;
    let mut output = 0.0;

    if fir_b.len() > 1 {
        output += process_fir_sample(&fir_b, input, &mut fir_state, &mut fir_index, channel)?;
    } else if fir_b.len() == 1 {
        output += fir_b[0] * input;
    }

    if delay_b.len() > 1 {
        input = process_fir_sample(&delay_b, input, &mut delay_state, &mut delay_index, channel)?;
    }

    for section in 0..iir_b.shape()[0] {
        let section_output = iir_b[(section, 0)] * input + iir_state[(section, 0, channel)];
        iir_state[(section, 0, channel)] = iir_b[(section, 1)] * input
            - iir_a[(section, 1)] * section_output
            + iir_state[(section, 1, channel)];
        iir_state[(section, 1, channel)] =
            iir_b[(section, 2)] * input - iir_a[(section, 2)] * section_output;
        output += section_output;
    }

    Ok(output)
}

fn process_fir_sample(
    coefficients: &ndarray::ArrayView1<'_, f64>,
    input: f64,
    state: &mut ndarray::ArrayViewMut2<'_, f64>,
    index: &mut ndarray::ArrayViewMut1<'_, i64>,
    channel: usize,
) -> PyResult<f64> {
    let order = coefficients.len() - 1;
    if index[channel] < 0 || index[channel] as usize >= order {
        return Err(PyValueError::new_err(
            "FIR state index must point inside the circular state",
        ));
    }
    let write_index = index[channel] as usize;
    let mut output = coefficients[0] * input;
    for coefficient in 0..order {
        let read_index = (write_index + order - coefficient) % order;
        output += state[(read_index, channel)] * coefficients[coefficient + 1];
    }
    let next_write_index = (write_index + 1) % order;
    state[(next_write_index, channel)] = input;
    index[channel] = next_write_index as i64;
    Ok(output)
}

#[pyfunction]
fn fir_filtering_sample(
    b: PyReadonlyArray1<'_, f64>,
    input: f64,
    mut state: PyReadwriteArray2<'_, f64>,
    mut current_state_ind: PyReadwriteArray1<'_, i64>,
    channel: usize,
) -> PyResult<f64> {
    let b = b.as_array();
    let mut state = state.as_array_mut();
    let mut current_state_ind = current_state_ind.as_array_mut();

    if b.is_empty() {
        return Err(PyValueError::new_err("FIR coefficients cannot be empty"));
    }
    if state.shape()[0] != b.len() - 1 {
        return Err(PyValueError::new_err(
            "state length must match FIR filter order",
        ));
    }
    if state.shape()[1] <= channel || current_state_ind.len() <= channel {
        return Err(PyValueError::new_err(
            "filter state does not contain the requested channel",
        ));
    }

    let order = b.len() - 1;
    if order == 0 {
        return Ok(b[0] * input);
    }

    let write_index = current_state_ind[channel];
    if write_index < 0 || write_index as usize >= order {
        return Err(PyValueError::new_err(
            "FIR state index must point inside the circular state",
        ));
    }
    let write_index = write_index as usize;

    let mut output = b[0] * input;
    for coefficient in 0..order {
        let read_index = (write_index + order - coefficient) % order;
        output += state[(read_index, channel)] * b[coefficient + 1];
    }

    let next_write_index = (write_index + 1) % order;
    state[(next_write_index, channel)] = input;
    current_state_ind[channel] = next_write_index as i64;
    Ok(output)
}

#[pyfunction]
fn iir_filtering_sample(
    b: PyReadonlyArray1<'_, f64>,
    a: PyReadonlyArray1<'_, f64>,
    input: f64,
    mut state: PyReadwriteArray2<'_, f64>,
    channel: usize,
) -> PyResult<f64> {
    let b = b.as_array();
    let a = a.as_array();
    let mut state = state.as_array_mut();

    if b.is_empty() || b.len() != a.len() {
        return Err(PyValueError::new_err(
            "IIR coefficient vectors must be non-empty and equally sized",
        ));
    }
    if state.shape()[0] != b.len() - 1 || state.shape()[1] <= channel {
        return Err(PyValueError::new_err(
            "IIR state shape does not match the filter order and channel",
        ));
    }

    let order = b.len() - 1;
    if order == 0 {
        return Ok(b[0] * input);
    }

    let output = b[0] * input + state[(0, channel)];
    for coefficient in 0..order - 1 {
        state[(coefficient, channel)] = input * b[coefficient + 1] - output * a[coefficient + 1]
            + state[(coefficient + 1, channel)];
    }
    state[(order - 1, channel)] = input * b[order] - output * a[order];
    Ok(output)
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

#[pyfunction]
fn lattice_filtering_fir_sample(
    k: PyReadonlyArray1<'_, f64>,
    input: f64,
    mut state: PyReadwriteArray2<'_, f64>,
    channel: usize,
) -> PyResult<f64> {
    let k = k.as_array();
    let mut state = state.as_array_mut();
    validate_lattice_fir_state(state.shape(), k.len(), channel + 1)?;

    let mut output = input;
    let mut previous = output;
    for coefficient in 0..k.len() {
        let next = -output * k[coefficient] + state[(coefficient, channel)];
        output -= state[(coefficient, channel)] * k[coefficient];
        state[(coefficient, channel)] = previous;
        previous = next;
    }

    Ok(output)
}

#[pyfunction]
fn lattice_filtering_iir(
    k: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    mut td: PyReadwriteArray2<'_, f64>,
    mut state: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    let k = k.as_array();
    let c = c.as_array();
    let mut td = td.as_array_mut();
    let mut state = state.as_array_mut();

    if k.is_empty() || c.len() != k.len() + 1 {
        return Err(PyValueError::new_err(
            "ladder coefficients must have one more value than reflection coefficients",
        ));
    }
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

    let order_iterations = k.len() - 1;
    for channel in 0..td.shape()[1] {
        for sample in 0..td.shape()[0] {
            let mut input = td[(sample, channel)];
            let mut low_output = 0.0;

            for coefficient in (0..=order_iterations).rev() {
                input += state[(coefficient, channel)] * k[coefficient];
                let section_output = input * -k[coefficient] + state[(coefficient, channel)];
                if coefficient != order_iterations {
                    state[(coefficient + 1, channel)] = section_output;
                }
                low_output += section_output * c[coefficient + 1];
            }

            state[(0, channel)] = input;
            td[(sample, channel)] = input * c[0] + low_output;
        }
    }

    Ok(())
}

#[pyfunction]
fn lattice_filtering_iir_sample(
    k: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    input: f64,
    mut state: PyReadwriteArray2<'_, f64>,
    channel: usize,
) -> PyResult<f64> {
    let k = k.as_array();
    let c = c.as_array();
    let mut state = state.as_array_mut();
    validate_lattice_iir_state(state.shape(), k.len(), c.len(), channel + 1)?;

    let mut input = input;
    let mut low_output = 0.0;
    for coefficient in (0..k.len()).rev() {
        input += state[(coefficient, channel)] * k[coefficient];
        let section_output = input * -k[coefficient] + state[(coefficient, channel)];
        if coefficient != k.len() - 1 {
            state[(coefficient + 1, channel)] = section_output;
        }
        low_output += section_output * c[coefficient + 1];
    }
    state[(0, channel)] = input;

    Ok(input * c[0] + low_output)
}

#[pyfunction]
fn lattice_filtering_sos(
    k: PyReadonlyArray2<'_, f64>,
    c: PyReadonlyArray2<'_, f64>,
    mut td: PyReadwriteArray2<'_, f64>,
    mut state: PyReadwriteArray3<'_, f64>,
) -> PyResult<()> {
    let k = k.as_array();
    let c = c.as_array();
    let mut td = td.as_array_mut();
    let mut state = state.as_array_mut();

    if k.shape()[1] != 2 || c.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "SOS coefficients must have shapes (sections, 2) and (sections, 3)",
        ));
    }
    if c.shape()[0] != k.shape()[0] || state.shape()[0] != k.shape()[0] {
        return Err(PyValueError::new_err(
            "SOS coefficient and state section counts must match",
        ));
    }
    if state.shape()[1] != 2 || state.shape()[2] != td.shape()[1] {
        return Err(PyValueError::new_err(
            "SOS state must have shape (sections, 2, channels)",
        ));
    }

    for channel in 0..td.shape()[1] {
        for sample in 0..td.shape()[0] {
            let mut input = td[(sample, channel)];
            for section in 0..k.shape()[0] {
                let mut low_output = 0.0;

                input += state[(section, 1, channel)] * k[(section, 1)];
                let section_output = input * -k[(section, 1)] + state[(section, 1, channel)];
                low_output += section_output * c[(section, 2)];

                input += state[(section, 0, channel)] * k[(section, 0)];
                let section_output = input * -k[(section, 0)] + state[(section, 0, channel)];
                state[(section, 1, channel)] = section_output;
                low_output += section_output * c[(section, 1)];
                state[(section, 0, channel)] = input;

                input = input * c[(section, 0)] + low_output;
            }
            td[(sample, channel)] = input;
        }
    }

    Ok(())
}

#[pyfunction]
fn lattice_filtering_sos_sample(
    k: PyReadonlyArray2<'_, f64>,
    c: PyReadonlyArray2<'_, f64>,
    input: f64,
    mut state: PyReadwriteArray3<'_, f64>,
    channel: usize,
) -> PyResult<f64> {
    let k = k.as_array();
    let c = c.as_array();
    let mut state = state.as_array_mut();
    validate_lattice_sos_state(state.shape(), k.shape(), c.shape(), channel + 1)?;

    let mut input = input;
    for section in 0..k.shape()[0] {
        let mut low_output = 0.0;

        input += state[(section, 1, channel)] * k[(section, 1)];
        let section_output = input * -k[(section, 1)] + state[(section, 1, channel)];
        low_output += section_output * c[(section, 2)];

        input += state[(section, 0, channel)] * k[(section, 0)];
        let section_output = input * -k[(section, 0)] + state[(section, 0, channel)];
        state[(section, 1, channel)] = section_output;
        low_output += section_output * c[(section, 1)];
        state[(section, 0, channel)] = input;

        input = input * c[(section, 0)] + low_output;
    }

    Ok(input)
}

fn validate_lattice_fir_state(
    state_shape: &[usize],
    coefficient_count: usize,
    n_channels: usize,
) -> PyResult<()> {
    if state_shape[0] != coefficient_count {
        return Err(PyValueError::new_err(
            "state length must match filter order",
        ));
    }
    if state_shape[1] < n_channels {
        return Err(PyValueError::new_err(
            "state channels must include the requested channel",
        ));
    }
    Ok(())
}

fn validate_lattice_iir_state(
    state_shape: &[usize],
    coefficient_count: usize,
    ladder_count: usize,
    n_channels: usize,
) -> PyResult<()> {
    if coefficient_count == 0 || ladder_count != coefficient_count + 1 {
        return Err(PyValueError::new_err(
            "ladder coefficients must have one more value than reflection coefficients",
        ));
    }
    validate_lattice_fir_state(state_shape, coefficient_count, n_channels)
}

fn validate_lattice_sos_state(
    state_shape: &[usize],
    k_shape: &[usize],
    c_shape: &[usize],
    n_channels: usize,
) -> PyResult<()> {
    if k_shape[1] != 2 || c_shape[1] != 3 {
        return Err(PyValueError::new_err(
            "SOS coefficients must have shapes (sections, 2) and (sections, 3)",
        ));
    }
    if c_shape[0] != k_shape[0] || state_shape[0] != k_shape[0] {
        return Err(PyValueError::new_err(
            "SOS coefficient and state section counts must match",
        ));
    }
    if state_shape[1] != 2 || state_shape[2] < n_channels {
        return Err(PyValueError::new_err(
            "SOS state must have shape (sections, 2, channels)",
        ));
    }
    Ok(())
}

#[pyfunction]
fn warped_fir_filtering(
    b: PyReadonlyArray1<'_, f64>,
    warp: f64,
    mut td: PyReadwriteArray2<'_, f64>,
    mut state: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    let b = b.as_array();
    let mut td = td.as_array_mut();
    let mut state = state.as_array_mut();
    validate_warped_state(state.shape(), b.len(), td.shape()[1])?;

    for channel in 0..td.shape()[1] {
        for sample in 0..td.shape()[0] {
            td[(sample, channel)] =
                process_warped_fir_sample(&b, warp, td[(sample, channel)], &mut state, channel);
        }
    }

    Ok(())
}

#[pyfunction]
fn warped_fir_filtering_sample(
    b: PyReadonlyArray1<'_, f64>,
    warp: f64,
    input: f64,
    mut state: PyReadwriteArray2<'_, f64>,
    channel: usize,
) -> PyResult<f64> {
    let b = b.as_array();
    let mut state = state.as_array_mut();
    validate_warped_state(state.shape(), b.len(), channel + 1)?;
    Ok(process_warped_fir_sample(
        &b, warp, input, &mut state, channel,
    ))
}

#[pyfunction]
fn warped_fir_filtering_block(
    b: PyReadonlyArray1<'_, f64>,
    warp: f64,
    input: PyReadonlyArray1<'_, f64>,
    mut output: PyReadwriteArray1<'_, f64>,
    mut state: PyReadwriteArray2<'_, f64>,
    channel: usize,
) -> PyResult<()> {
    let b = b.as_array();
    let input = input.as_array();
    let mut output = output.as_array_mut();
    let mut state = state.as_array_mut();
    validate_warped_state(state.shape(), b.len(), channel + 1)?;
    if input.len() != output.len() {
        return Err(PyValueError::new_err("input and output lengths must match"));
    }

    for sample in 0..input.len() {
        output[sample] = process_warped_fir_sample(&b, warp, input[sample], &mut state, channel);
    }

    Ok(())
}

#[pyfunction]
fn warped_iir_filtering(
    b: PyReadonlyArray1<'_, f64>,
    sigmas: PyReadonlyArray1<'_, f64>,
    warp: f64,
    mut td: PyReadwriteArray2<'_, f64>,
    mut state: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    let b = b.as_array();
    let sigmas = sigmas.as_array();
    let mut td = td.as_array_mut();
    let mut state = state.as_array_mut();
    validate_warped_iir_state(state.shape(), b.len(), sigmas.len(), td.shape()[1])?;

    for channel in 0..td.shape()[1] {
        for sample in 0..td.shape()[0] {
            td[(sample, channel)] = process_warped_iir_sample(
                &b,
                &sigmas,
                warp,
                td[(sample, channel)],
                &mut state,
                channel,
            );
        }
    }

    Ok(())
}

#[pyfunction]
fn warped_iir_filtering_sample(
    b: PyReadonlyArray1<'_, f64>,
    sigmas: PyReadonlyArray1<'_, f64>,
    warp: f64,
    input: f64,
    mut state: PyReadwriteArray2<'_, f64>,
    channel: usize,
) -> PyResult<f64> {
    let b = b.as_array();
    let sigmas = sigmas.as_array();
    let mut state = state.as_array_mut();
    validate_warped_iir_state(state.shape(), b.len(), sigmas.len(), channel + 1)?;
    Ok(process_warped_iir_sample(
        &b, &sigmas, warp, input, &mut state, channel,
    ))
}

#[pyfunction]
fn warped_iir_filtering_block(
    b: PyReadonlyArray1<'_, f64>,
    sigmas: PyReadonlyArray1<'_, f64>,
    warp: f64,
    input: PyReadonlyArray1<'_, f64>,
    mut output: PyReadwriteArray1<'_, f64>,
    mut state: PyReadwriteArray2<'_, f64>,
    channel: usize,
) -> PyResult<()> {
    let b = b.as_array();
    let sigmas = sigmas.as_array();
    let input = input.as_array();
    let mut output = output.as_array_mut();
    let mut state = state.as_array_mut();
    validate_warped_iir_state(state.shape(), b.len(), sigmas.len(), channel + 1)?;
    if input.len() != output.len() {
        return Err(PyValueError::new_err("input and output lengths must match"));
    }

    for sample in 0..input.len() {
        output[sample] =
            process_warped_iir_sample(&b, &sigmas, warp, input[sample], &mut state, channel);
    }

    Ok(())
}

fn validate_warped_state(
    state_shape: &[usize],
    coefficient_count: usize,
    n_channels: usize,
) -> PyResult<()> {
    if coefficient_count == 0 {
        return Err(PyValueError::new_err("coefficients cannot be empty"));
    }
    if state_shape[0] != coefficient_count {
        return Err(PyValueError::new_err(
            "state length must match FIR coefficient count",
        ));
    }
    if state_shape[1] < n_channels {
        return Err(PyValueError::new_err(
            "state channels must include the requested channels",
        ));
    }
    Ok(())
}

fn validate_warped_iir_state(
    state_shape: &[usize],
    coefficient_count: usize,
    sigma_count: usize,
    n_channels: usize,
) -> PyResult<()> {
    if coefficient_count == 0 || sigma_count == 0 {
        return Err(PyValueError::new_err("coefficients cannot be empty"));
    }
    if state_shape[0] < coefficient_count || state_shape[0] < sigma_count - 1 {
        return Err(PyValueError::new_err(
            "state length must cover the warped IIR coefficients",
        ));
    }
    if state_shape[1] < n_channels {
        return Err(PyValueError::new_err(
            "state channels must include the requested channels",
        ));
    }
    Ok(())
}

fn process_warped_fir_sample(
    b: &ndarray::ArrayView1<'_, f64>,
    warp: f64,
    input: f64,
    state: &mut ndarray::ArrayViewMut2<'_, f64>,
    channel: usize,
) -> f64 {
    let mut output = input * b[0];
    let mut residue = input;
    let order = state.shape()[0] - 1;

    for coefficient in 0..order {
        let new_residue =
            (state[(coefficient + 1, channel)] - residue) * warp + state[(coefficient, channel)];
        state[(coefficient, channel)] = residue;
        residue = new_residue;
        if coefficient + 1 < b.len() {
            output += new_residue * b[coefficient + 1];
        }
    }

    state[(order, channel)] = residue;
    output
}

fn process_warped_iir_sample(
    b: &ndarray::ArrayView1<'_, f64>,
    sigmas: &ndarray::ArrayView1<'_, f64>,
    warp: f64,
    mut input: f64,
    state: &mut ndarray::ArrayViewMut2<'_, f64>,
    channel: usize,
) -> f64 {
    for sigma in 1..sigmas.len() {
        input += sigmas[sigma] * state[(sigma - 1, channel)];
    }
    input *= sigmas[0];
    process_warped_fir_sample(b, warp, input, state, channel)
}
