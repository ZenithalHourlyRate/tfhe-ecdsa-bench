//! Small conversion helpers between `num_bigint::BigInt` and tfhe-rs radix types.
//!
//! Ported from the original TFHECDSA repo (`src/helper.rs`) to tfhe-rs 1.8.1.

use num_bigint::{BigInt, Sign};
use tfhe::integer::block_decomposition::{BlockDecomposer, DecomposableInto, RecomposableFrom};
use tfhe::integer::{U256, U512};

/// ceil(log2(value)), with the same conventions as the original repo.
pub fn bigint_ilog2_ceil(value: &BigInt) -> u32 {
    let mut value = value.clone();
    let mut i = 0;
    if value == BigInt::from(0) {
        0
    } else if value == BigInt::from(1) {
        1
    } else {
        while value > BigInt::from(0) {
            value >>= 1;
            i += 1;
        }
        i
    }
}

/// Interpret a tfhe radix scalar as a little-endian `BigInt`.
pub fn to_bigint<T: DecomposableInto<u8>>(a: T) -> BigInt {
    BigInt::from_bytes_le(
        Sign::Plus,
        &BlockDecomposer::new(a, 8)
            .iter_as::<u8>()
            .collect::<Vec<_>>(),
    )
}

/// Rebuild a tfhe radix scalar from a little-endian `BigInt` (truncating to `T::BITS`).
pub fn from_bigint<T: DecomposableInto<u8> + RecomposableFrom<u8>>(a: &BigInt) -> T {
    let mut res = T::ZERO;
    for (i, b) in a.to_bytes_le().1.iter().enumerate() {
        res += T::cast_from(*b) << (i * 8) as u32;
    }
    res
}

pub fn bigint_to_u128(a: &BigInt) -> u128 {
    let mut res = 0u128;
    for (i, b) in a.to_bytes_le().1.iter().enumerate() {
        res += (*b as u128) << (i * 8);
    }
    res
}

pub fn bigint_to_u256(a: &BigInt) -> U256 {
    from_bigint::<U256>(a)
}

pub fn bigint_to_u512(a: &BigInt) -> U512 {
    from_bigint::<U512>(a)
}
