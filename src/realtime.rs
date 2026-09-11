use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn add_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
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
