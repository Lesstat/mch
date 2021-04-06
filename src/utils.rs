use crate::F64_SIZE;

#[macro_export]
macro_rules! float_eq {
    ($lhs:expr, $rhs:expr) => {
        approx::abs_diff_eq!($lhs, $rhs, epsilon = $crate::ACCURACY)
    };
}

pub fn equal_weights(dim: usize) -> Vec<f64> {
    vec![1.0 / dim as f64; dim]
}

pub fn same_array(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| float_eq!(a, b))
}

pub fn convert_to_f64_vec(buffer: &mut [u8]) -> Vec<f64> {
    let mut byte_buffer = [0u8; F64_SIZE];
    buffer
        .chunks_exact(F64_SIZE)
        .map(|slice| {
            byte_buffer.copy_from_slice(slice);
            f64::from_ne_bytes(byte_buffer)
        })
        .collect()
}

pub fn add_edge_costs(a: &mut [f64], b: &[f64]) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a += b)
}

pub fn costs_by_alpha(costs: &[f64], alpha: &[f64]) -> f64 {
    assert_eq!(costs.len(), alpha.len());
    // The unsafe version of this hot loop safes a lot of runtime ...
    // The idiomatic version with iterators was about 10% slower :(
    let mut res = 0.0;
    for i in 0..costs.len() {
        // SAFETY: By above assert costs and alpha have the same length and get
        // accessed only on valid indices.
        unsafe {
            res += costs.get_unchecked(i) * alpha.get_unchecked(i);
        }
    }
    res
}
