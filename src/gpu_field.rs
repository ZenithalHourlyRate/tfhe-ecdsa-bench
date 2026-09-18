//! GPU (CUDA) port of the finite-field modular multiplication algorithms in
//! `field.rs`, built on the tfhe-rs 1.8.1 integer GPU backend (`CudaServerKey`).
//!
//! The algorithms are identical to the CPU versions; only the API surface
//! differs: every operation takes a `&CudaStreams`, works on
//! `CudaUnsignedRadixCiphertext` and returns a new ciphertext (the GPU backend
//! mostly exposes non-assigning variants of the operations used in `field.rs`).
//!
//! As on the CPU, `add`/`sub`/`mul` require both operands to have the same
//! number of blocks, hence the explicit trims/extensions below.

use num_bigint::BigInt;
use tfhe::core_crypto::gpu::CudaStreams;
use tfhe::integer::gpu::ciphertext::boolean_value::CudaBooleanBlock;
use tfhe::integer::gpu::ciphertext::{CudaIntegerRadixCiphertext, CudaUnsignedRadixCiphertext};
use tfhe::integer::gpu::CudaServerKey;
use tfhe::integer::{U256, U512};

use crate::helper::{bigint_ilog2_ceil, bigint_to_u128, from_bigint, to_bigint};

/// Number of radix blocks of a GPU ciphertext.
///
/// The GPU backend does not expose `blocks()`; `CudaLweCiphertextList::lwe_ciphertext_count`
/// is crate-private but `CudaRadixCiphertext::info` carries the same count.
pub fn blocks(ct: &CudaUnsignedRadixCiphertext) -> usize {
    ct.as_ref().info.blocks.len()
}

/// `selector ? a : 0` for a constant `a`, where `selector` encrypts one bit.
pub fn selector_zero_constant<const NB: usize>(
    a: U256,
    selector: &CudaBooleanBlock,
    server_key: &CudaServerKey,
    streams: &CudaStreams,
) -> CudaUnsignedRadixCiphertext {
    // The GPU backend has no `BooleanBlock::into_radix`; extending the 1-block
    // boolean with trivial zero MSB blocks gives an NB-block ciphertext of value
    // 0 or 1, and `scalar_mul` by `a` then yields 0 or `a` in NB blocks.
    let extended =
        server_key.extend_radix_with_trivial_zero_blocks_msb(selector.as_ref(), NB - 1, streams);
    server_key.scalar_mul(&extended, a, streams)
}

/// Fast reduction `x mod p` assuming `x < 2*p`. `x` must have at most `NB + 1`
/// blocks; the result is trimmed back to `NB` blocks.
pub fn modulo_fast<const NB: usize>(
    x: &CudaUnsignedRadixCiphertext,
    p: U256,
    server_key: &CudaServerKey,
    streams: &CudaStreams,
) -> CudaUnsignedRadixCiphertext {
    // The GPU scalar comparison requires an even number of blocks (or exactly
    // one); `x` has NB or NB + 1 blocks and NB is even, so at most one trivial
    // zero MSB block is needed to make the width even.
    let mut x = x.duplicate(streams);
    let mut len = blocks(&x);
    if len % 2 != 0 {
        x = server_key.extend_radix_with_trivial_zero_blocks_msb(&x, 1, streams);
        len += 1;
    }
    let is_ge = server_key.scalar_ge(&x, p, streams);
    let mut to_sub = selector_zero_constant::<NB>(p, &is_ge, server_key, streams);
    // GPU add/sub need matching widths; `to_sub` has NB blocks, `x` has `len`.
    if len > NB {
        to_sub =
            server_key.extend_radix_with_trivial_zero_blocks_msb(&to_sub, len - NB, streams);
    }
    server_key.sub_assign(&mut x, &to_sub, streams);
    server_key.trim_radix_blocks_msb(&x, len - NB, streams)
}

/// Full `NB`-block x `NB`-block ciphertext multiplication.
///
/// Both operands are zero-extended to `2*NB` blocks before multiplying: the
/// GPU schoolbook multiplication returns the low `lhs.blocks()` blocks of the
/// product and the `2*b`-bit product of two `b`-bit (zero-extended) operands
/// fits exactly in those `2*NB` blocks.
pub fn mul_wide<const NB: usize>(
    a: &CudaUnsignedRadixCiphertext,
    b: &CudaUnsignedRadixCiphertext,
    server_key: &CudaServerKey,
    streams: &CudaStreams,
) -> CudaUnsignedRadixCiphertext {
    let a_expanded = server_key.extend_radix_with_trivial_zero_blocks_msb(a, NB, streams);
    let b_expanded = server_key.extend_radix_with_trivial_zero_blocks_msb(b, NB, streams);
    server_key.mul(&a_expanded, &b_expanded, streams)
}

/// Barrett reduction of a 2*b-bit ciphertext `x` modulo `p`.
///
/// `m = floor(2^k / p)` with `k = 2 * (bits per block * NB)`: `q = (x*m) >> k`,
/// `x - q*p`, then one `modulo_fast`. `block_to_add` widens `x` so the low
/// blocks kept by `scalar_mul` still hold the full product `x*m`.
pub fn mod_barrett<const NB: usize>(
    x: &CudaUnsignedRadixCiphertext,
    p: U256,
    server_key: &CudaServerKey,
    streams: &CudaStreams,
) -> CudaUnsignedRadixCiphertext {
    let k = 4 * NB;
    let m_bigint = BigInt::from(2).pow(k as u32) / to_bigint(p);
    let block_to_add = (m_bigint.bits() - 2 * NB as u64 + 1) / 2;
    let m: U512 = from_bigint(&m_bigint);
    let x =
        server_key.extend_radix_with_trivial_zero_blocks_msb(x, NB + block_to_add as usize, streams);
    let q = server_key.scalar_mul(&x, m, streams);
    let q = server_key.scalar_right_shift(&q, k as u64, streams);
    let mut x = x;
    server_key.sub_assign(&mut x, &server_key.scalar_mul(&q, p, streams), streams);
    let len = blocks(&x);
    let x = server_key.trim_radix_blocks_msb(&x, len - (NB + 1), streams);
    modulo_fast::<NB>(&x, p, server_key, streams)
}

/// Decompose `p = 2^n - c`; `c` must satisfy `0 <= c <= 2^floor(n/2)` for the
/// Mersenne reduction below to work.
pub fn mersenne_coeff_p(p: U256) -> (u32, BigInt) {
    let pb = to_bigint(p);
    let n = bigint_ilog2_ceil(&pb);
    let c = (BigInt::from(1) << n) - &pb;
    (n, c)
}

/// `x mod p` for `x < p^2`, using two passes of Pseudo-Mersenne reduction
/// (`x = a*2^n + b  =>  x == c*a + b (mod p)`) followed by `modulo_fast`.
///
/// Falls back to Barrett when `c` is too large for the Mersenne trick
/// (secp256k1 group order `n`).
pub fn mod_mersenne<const NB: usize>(
    x: &CudaUnsignedRadixCiphertext,
    p: U256,
    server_key: &CudaServerKey,
    streams: &CudaStreams,
) -> CudaUnsignedRadixCiphertext {
    let (n, c) = mersenne_coeff_p(p);
    let ceilc = bigint_ilog2_ceil(&c);
    if ceilc >= n / 2 {
        return mod_barrett::<NB>(x, p, server_key, streams);
    }
    let c_blocks = (c.bits() as usize + 1) / 2;
    let x = server_key.extend_radix_with_trivial_zero_blocks_msb(x, (NB * 2) - blocks(x), streams);

    // first pass: NB*2 blocks
    let x_mod_p = {
        let a = server_key.scalar_right_shift(&x, n as u64, streams);
        let b = server_key.sub(&x, &server_key.scalar_left_shift(&a, n as u64, streams), streams);

        let len = blocks(&x);
        // `a` is multiplied by `c`, so it must be at least NB + c_blocks long;
        // `b` is widened to the same length for the addition below.
        let a = server_key.trim_radix_blocks_msb(&a, len - (NB + c_blocks), streams);
        let b = server_key.trim_radix_blocks_msb(&b, len - NB, streams);
        let ca = server_key.scalar_mul(&a, bigint_to_u128(&c), streams);
        let b = server_key.extend_radix_with_trivial_zero_blocks_msb(&b, c_blocks, streams);
        server_key.add(&ca, &b, streams)
    };

    // second pass: NB + c_blocks blocks
    let x_mod_p2 = {
        let a = server_key.scalar_right_shift(&x_mod_p, n as u64, streams);
        let b =
            server_key.sub(&x_mod_p, &server_key.scalar_left_shift(&a, n as u64, streams), streams);

        let len = blocks(&x_mod_p);
        // `a` is multiplied by `c`, so it must be at least NB + 1 long; `b` is
        // widened to the same length for the addition below.
        let a = server_key.trim_radix_blocks_msb(&a, len - (NB + 1), streams);
        let b = server_key.trim_radix_blocks_msb(&b, len - NB, streams);
        let ca = server_key.scalar_mul(&a, bigint_to_u128(&c), streams);
        let b = server_key.extend_radix_with_trivial_zero_blocks_msb(&b, 1, streams);
        server_key.add(&ca, &b, streams)
    };

    modulo_fast::<NB>(&x_mod_p2, p, server_key, streams)
}

/// `a * b mod p` with a 2*b-bit product and the Pseudo-Mersenne / Barrett
/// reduction above.
pub fn mul_mod_mersenne<const NB: usize>(
    a: &CudaUnsignedRadixCiphertext,
    b: &CudaUnsignedRadixCiphertext,
    p: U256,
    server_key: &CudaServerKey,
    streams: &CudaStreams,
) -> CudaUnsignedRadixCiphertext {
    let x = mul_wide::<NB>(a, b, server_key, streams);
    mod_mersenne::<NB>(&x, p, server_key, streams)
}

/// Field multiplication dispatcher (Mersenne path for F_p, Barrett for F_n).
pub fn mul_mod<const NB: usize>(
    a: &CudaUnsignedRadixCiphertext,
    b: &CudaUnsignedRadixCiphertext,
    p: U256,
    server_key: &CudaServerKey,
    streams: &CudaStreams,
) -> CudaUnsignedRadixCiphertext {
    mul_mod_mersenne::<NB>(a, b, p, server_key, streams)
}

/// Naive baseline: wide multiply followed by an arithmetic scalar division to
/// obtain the remainder (`server_key.scalar_div_rem`).
pub fn mul_mod_naive_div_rem<const NB: usize>(
    a: &CudaUnsignedRadixCiphertext,
    b: &CudaUnsignedRadixCiphertext,
    p: U256,
    server_key: &CudaServerKey,
    streams: &CudaStreams,
) -> CudaUnsignedRadixCiphertext {
    let x = mul_wide::<NB>(a, b, server_key, streams);
    // As on the CPU, the scalar divisor type must cover all encrypted bits of
    // the numerator (2*NB*2 = 512 bits here), so a 512-bit scalar is used even
    // though the modulus itself is 256-bit.
    let p512: U512 = from_bigint(&to_bigint(p));
    let (_q, r) = server_key.scalar_div_rem(&x, p512, streams);
    let len = blocks(&r);
    server_key.trim_radix_blocks_msb(&r, len - NB, streams)
}
