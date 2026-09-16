//! Finite-field modular multiplication for the secp256k1 base field F_p and
//! scalar field F_n, implemented on top of tfhe-rs 1.8.1 integer primitives.
//!
//! tfhe-rs does **not** provide a finite-field multiplication API, so this is a
//! port of the algorithms from the original TFHECDSA repo:
//!   * `mul_mod_mersenne`: 2*b-bit (ciphertext) * b-bit (ciphertext) multiply
//!     followed by a Pseudo-Mersenne reduction for p (p = 2^256 - k, small k),
//!     and Barrett reduction as fallback when k is too large (which is the case
//!     for the secp256k1 group order n).

use num_bigint::BigInt;
use tfhe::integer::{BooleanBlock, IntegerCiphertext, RadixCiphertext, ServerKey, U256, U512};

use crate::helper::{bigint_ilog2_ceil, bigint_to_u128, from_bigint, to_bigint};

/// `selector ? a : 0` for a constant `a`, where `selector` encrypts one bit.
///
/// The result is a `NB`-block radix ciphertext.
pub fn selector_zero_constant<const NB: usize>(
    a: U256,
    selector: &BooleanBlock,
    server_key: &ServerKey,
) -> RadixCiphertext {
    // In tfhe-rs >= 0.4 comparators return a `BooleanBlock`; `into_radix`
    // broadcasts the encrypted bit into `NB` blocks (trivial zero blocks on top).
    let mut selector: RadixCiphertext = selector.clone().into_radix(NB, server_key);
    server_key.scalar_mul_assign_parallelized(&mut selector, a);
    selector
}

/// Fast reduction `x mod p` assuming `x < 2*p`. `x` must have at most `NB + 1`
/// blocks; the result is trimmed back to `NB` blocks.
pub fn modulo_fast<const NB: usize>(
    x: &RadixCiphertext,
    p: U256,
    server_key: &ServerKey,
) -> RadixCiphertext {
    let len = x.blocks().len();
    let mut x = x.clone();
    let is_gt = server_key.scalar_ge_parallelized(&x, p);
    let mut to_sub = selector_zero_constant::<NB>(p, &is_gt, server_key);
    // tfhe-rs >= 0.4 requires both operands of add/sub to have the same length.
    let missing = len - to_sub.blocks().len();
    if missing > 0 {
        server_key.extend_radix_with_trivial_zero_blocks_msb_assign(&mut to_sub, missing);
    }
    server_key.sub_assign_parallelized(&mut x, &to_sub);
    server_key.trim_radix_blocks_msb_assign(&mut x, len - NB);
    x
}

/// Full `NB`-block x `NB`-block ciphertext multiplication.
///
/// tfhe-rs 1.8.1 only computes the correct low part of a product when
/// `rhs` is at least as wide as `lhs` (the carry of the most significant
/// `rhs` block is dropped otherwise). We therefore zero-extend **both**
/// operands to `2*NB` blocks; the exact `2*b`-bit product fits in the low
/// `2*NB` blocks, so the result is exact.
pub fn mul_wide<const NB: usize>(
    a: &RadixCiphertext,
    b: &RadixCiphertext,
    server_key: &ServerKey,
) -> RadixCiphertext {
    let mut a_expanded = server_key.extend_radix_with_trivial_zero_blocks_msb(a, NB);
    let b_expanded = server_key.extend_radix_with_trivial_zero_blocks_msb(b, NB);
    server_key.mul_assign_parallelized(&mut a_expanded, &b_expanded);
    let len = a_expanded.blocks().len();
    if len > 2 * NB {
        server_key.trim_radix_blocks_msb_assign(&mut a_expanded, len - 2 * NB);
    }
    a_expanded
}

/// Barrett reduction of a 2*b-bit ciphertext `x` modulo `p`.
///
/// `m = floor(2^k / p)` with `k = 2 * (bits per block * NB)` (i.e. `4*NB` for
/// 2-bit blocks): `q = (x*m) >> k`, `x - q*p`, then one `modulo_fast`.
pub fn mod_barrett<const NB: usize>(
    x: &RadixCiphertext,
    p: U256,
    server_key: &ServerKey,
) -> RadixCiphertext {
    let k = 4 * NB;
    let m_bigint = BigInt::from(2).pow(k as u32) / to_bigint(p);
    let block_to_add = (m_bigint.bits() - 2 * NB as u64 + 1) / 2;
    let m: U512 = from_bigint(&m_bigint);
    let mut x =
        server_key.extend_radix_with_trivial_zero_blocks_msb(x, NB + block_to_add as usize);
    let mut q = server_key.scalar_mul_parallelized(&x, m);
    server_key.scalar_right_shift_assign_parallelized(&mut q, k as u64);
    server_key.sub_assign_parallelized(&mut x, &server_key.scalar_mul_parallelized(&q, p));
    let len = x.blocks().len();
    server_key.trim_radix_blocks_msb_assign(&mut x, len - (NB + 1));
    modulo_fast::<NB>(&x, p, server_key)
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
    x: &RadixCiphertext,
    p: U256,
    server_key: &ServerKey,
) -> RadixCiphertext {
    let (n, c) = mersenne_coeff_p(p);
    let ceilc = bigint_ilog2_ceil(&c);
    if ceilc >= n / 2 {
        return mod_barrett::<NB>(x, p, server_key);
    }
    let c_blocks = (c.bits() as usize + 1) / 2;
    let x = server_key.extend_radix_with_trivial_zero_blocks_msb(x, (NB * 2) - x.blocks().len());

    // first pass: NB*2 blocks
    let x_mod_p = (|x: &RadixCiphertext| {
        let mut a = server_key.scalar_right_shift_parallelized(x, n as u64);
        let mut b = server_key
            .sub_parallelized(x, &server_key.scalar_left_shift_parallelized(&a, n as u64));

        let len = x.blocks().len();
        // a will be multiplied by c, so it must be at least NB + c_blocks long
        server_key.trim_radix_blocks_msb_assign(&mut a, len - (NB + c_blocks));
        // b must be at least NB long, and the add below requires matching widths
        server_key.trim_radix_blocks_msb_assign(&mut b, len - NB);
        let ca = server_key.scalar_mul_parallelized(&a, bigint_to_u128(&c));
        server_key.extend_radix_with_trivial_zero_blocks_msb_assign(&mut b, c_blocks);
        server_key.add_parallelized(&ca, &b)
    })(&x);

    // second pass: NB + c_blocks blocks
    let x_mod_p2 = (|x: &RadixCiphertext| {
        let mut a = server_key.scalar_right_shift_parallelized(x, n as u64);
        let mut b = server_key
            .sub_parallelized(x, &server_key.scalar_left_shift_parallelized(&a, n as u64));

        let len = x.blocks().len();
        // a will be multiplied by c, so it must be at least NB + 1 long
        server_key.trim_radix_blocks_msb_assign(&mut a, len - (NB + 1));
        // b must be at least NB long, and the add below requires matching widths
        server_key.trim_radix_blocks_msb_assign(&mut b, len - NB);
        let ca = server_key.scalar_mul_parallelized(&a, bigint_to_u128(&c));
        server_key.extend_radix_with_trivial_zero_blocks_msb_assign(&mut b, 1);
        server_key.add_parallelized(&ca, &b)
    })(&x_mod_p);

    modulo_fast::<NB>(&x_mod_p2, p, server_key)
}

/// `a * b mod p` with a 2*b-bit product and the Pseudo-Mersenne / Barrett
/// reduction above.
pub fn mul_mod_mersenne<const NB: usize>(
    a: &RadixCiphertext,
    b: &RadixCiphertext,
    p: U256,
    server_key: &ServerKey,
) -> RadixCiphertext {
    let x = mul_wide::<NB>(a, b, server_key);
    mod_mersenne::<NB>(&x, p, server_key)
}

/// Field multiplication dispatcher (Mersenne path for F_p, Barrett for F_n).
pub fn mul_mod<const NB: usize>(
    a: &RadixCiphertext,
    b: &RadixCiphertext,
    p: U256,
    server_key: &ServerKey,
) -> RadixCiphertext {
    mul_mod_mersenne::<NB>(a, b, p, server_key)
}

/// Naive baseline: wide multiply followed by an arithmetic scalar division to
/// obtain the remainder (`server_key.scalar_div_rem_parallelized`).
pub fn mul_mod_naive_div_rem<const NB: usize>(
    a: &RadixCiphertext,
    b: &RadixCiphertext,
    p: U256,
    server_key: &ServerKey,
) -> RadixCiphertext {
    let x = mul_wide::<NB>(a, b, server_key);
    // tfhe-rs 1.8.1 requires the scalar divisor type to cover all encrypted
    // bits of the numerator (2*NB*2 = 512 bits here), so a 512-bit scalar is
    // used even though the modulus itself is 256-bit.
    let p512: U512 = from_bigint(&to_bigint(p));
    let (_q, r) = server_key.scalar_div_rem_parallelized(&x, p512);
    let mut r = r;
    let len = r.blocks().len();
    server_key.trim_radix_blocks_msb_assign(&mut r, len - NB);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper::{bigint_to_u256, to_bigint};
    use num_bigint::BigInt;
    use tfhe::integer::keycache::IntegerKeyCache;
    use tfhe::integer::{ClientKey, IntegerKeyKind};

    /// Small-NB port validation (16-bit values) so the whole test runs in
    /// seconds instead of hours.
    const TNB: usize = 8;

    fn keys() -> (ClientKey, ServerKey) {
        IntegerKeyCache::new().get_from_params(
            tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2,
            IntegerKeyKind::Radix,
        )
    }

    fn enc(client_key: &ClientKey, value: u64) -> RadixCiphertext {
        client_key.encrypt_radix(value, TNB)
    }

    #[test]
    fn mul_wide_is_exact() {
        let (client_key, server_key) = keys();
        let a = 0xFFFFu64;
        let b = 0x1234u64;
        let product = mul_wide::<TNB>(&enc(&client_key, a), &enc(&client_key, b), &server_key);
        let dec: u32 = client_key.decrypt_radix(&product);
        assert_eq!(dec, (a * b) as u32);
    }

    #[test]
    fn mul_mod_mersenne_prime() {
        // 251 = 2^8 - 5 -> Pseudo-Mersenne path
        let (client_key, server_key) = keys();
        let p = 251u64;
        for (a, b) in [(250u64, 249u64), (1, 1), (123, 250), (17, 200)] {
            let out = mul_mod::<TNB>(
                &enc(&client_key, a),
                &enc(&client_key, b),
                bigint_to_u256(&BigInt::from(p)),
                &server_key,
            );
            let dec: u32 = client_key.decrypt_radix(&out);
            assert_eq!(dec, ((a * b) % p) as u32, "a={a} b={b}");
        }
    }

    #[test]
    fn mul_mod_barrett_prime() {
        // 263: c = 2^9 - 263 = 249 is too large -> Barrett path
        let (client_key, server_key) = keys();
        let p = 263u64;
        for (a, b) in [(262u64, 261u64), (1, 1), (123, 250), (17, 200)] {
            let out = mul_mod::<TNB>(
                &enc(&client_key, a),
                &enc(&client_key, b),
                bigint_to_u256(&BigInt::from(p)),
                &server_key,
            );
            let dec: u32 = client_key.decrypt_radix(&out);
            assert_eq!(dec, ((a * b) % p) as u32, "a={a} b={b}");
        }
    }

    #[test]
    fn barrett_and_mersenne_agree() {
        let (client_key, server_key) = keys();
        let p = 263u64;
        let a = 300u64;
        let b = 411u64; // a*b = 123300 > 2^16, exercise wide product
        let product = mul_wide::<TNB>(&enc(&client_key, a), &enc(&client_key, b), &server_key);
        let p_u256 = bigint_to_u256(&BigInt::from(p));
        let r_barrett = mod_barrett::<TNB>(&product, p_u256, &server_key);
        let r_mersenne = mod_mersenne::<TNB>(&product, p_u256, &server_key);
        let expected = ((a * b) % p) as u32;
        let d1: u32 = client_key.decrypt_radix(&r_barrett);
        let d2: u32 = client_key.decrypt_radix(&r_mersenne);
        assert_eq!(d1, expected);
        assert_eq!(d2, expected);
    }

    #[test]
    fn mod_mersenne_dispatch_small() {
        // mersenne_coeff for both primes
        let (n1, c1) = mersenne_coeff_p(bigint_to_u256(&BigInt::from(251u64)));
        assert_eq!((n1, c1), (8, BigInt::from(5)));
        let (n2, c2) = mersenne_coeff_p(bigint_to_u256(&BigInt::from(263u64)));
        assert_eq!((n2, c2), (9, BigInt::from(249)));
        assert_eq!(to_bigint(bigint_to_u256(&BigInt::from(263u64))), BigInt::from(263));
    }
}

#[cfg(test)]
mod diag_tests {
    use super::*;
    use crate::helper::{bigint_to_u256, bigint_to_u512, to_bigint};
    use num_bigint::BigInt;
    use tfhe::integer::keycache::IntegerKeyCache;
    use tfhe::integer::{ClientKey, IntegerKeyKind, U256};

    fn keys() -> (ClientKey, ServerKey) {
        IntegerKeyCache::new().get_from_params(
            tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2,
            IntegerKeyKind::Radix,
        )
    }

    const P_STR: &[u8] = b"115792089237316195423570985008687907853269984665640564039457584007908834671663";

    #[test]
    fn conversions_roundtrip_large() {
        let x = BigInt::parse_bytes(P_STR, 10).unwrap();
        assert_eq!(to_bigint(bigint_to_u256(&x)), x);
        let prod = &x * &x;
        assert_eq!(to_bigint(bigint_to_u512(&prod)), prod);
        let minus = &x - BigInt::from(12345u64);
        assert_eq!(to_bigint(bigint_to_u256(&minus)), minus);
    }

    #[test]
    fn encrypt_decrypt_large_value() {
        let (client_key, _server_key) = keys();
        let x = BigInt::parse_bytes(P_STR, 10).unwrap() - BigInt::from(1u64);
        let enc = client_key.encrypt_radix(bigint_to_u256(&x), 128);
        let dec: U256 = client_key.decrypt_radix(&enc);
        assert_eq!(to_bigint(dec), x);
    }

    #[test]
    fn mul_wide_nb_32() {
        const LNB: usize = 32;
        let (client_key, server_key) = keys();
        let a: u128 = 0x1234_5678_9abc_def0;
        let b: u128 = 0xfedc_ba98_7654_3210;
        let ea = client_key.encrypt_radix(a, LNB);
        let eb = client_key.encrypt_radix(b, LNB);
        let prod = mul_wide::<LNB>(&ea, &eb, &server_key);
        let dec: u128 = client_key.decrypt_radix(&prod);
        assert_eq!(dec, a * b, "nb=32 wide mul mismatch");
    }

    #[test]
    fn mul_wide_nb_64() {
        const LNB: usize = 64;
        let (client_key, server_key) = keys();
        let a: u128 = 0x1234_5678_9abc_def0_1111_2222_3333_4444;
        let b: u128 = 0xfedc_ba98_7654_3210_aaaa_bbbb_cccc_dddd;
        let ea = client_key.encrypt_radix(a, LNB);
        let eb = client_key.encrypt_radix(b, LNB);
        let prod = mul_wide::<LNB>(&ea, &eb, &server_key);
        let dec: U256 = client_key.decrypt_radix(&prod);
        assert_eq!(
            to_bigint(dec),
            BigInt::from(a) * BigInt::from(b),
            "nb=64 wide mul mismatch"
        );
    }

    #[test]
    fn mul_mod_nb_32_mersenne() {
        const LNB: usize = 32;
        let (client_key, server_key) = keys();
        let p: u128 = 0xFFFF_FFFF_FFFF_FFC5; // 2^64 - 59, prime
        let a: u128 = p - 2;
        let b: u128 = p - 3;
        let ea = client_key.encrypt_radix(a, LNB);
        let eb = client_key.encrypt_radix(b, LNB);
        let em = bigint_to_u256(&BigInt::from(p));
        let out = mul_mod::<LNB>(&ea, &eb, em, &server_key);
        let dec: u128 = client_key.decrypt_radix(&out);
        assert_eq!(dec, (a * b) % p, "nb=32 mul_mod mismatch");
    }
}

#[cfg(test)]
mod diag2 {
    use super::*;
    use tfhe::integer::keycache::IntegerKeyCache;
    use tfhe::integer::{ClientKey, IntegerKeyKind, U256};

    fn keys() -> (ClientKey, ServerKey) {
        IntegerKeyCache::new().get_from_params(
            tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2,
            IntegerKeyKind::Radix,
        )
    }

    #[test]
    fn native_same_size_mul_nb32() {
        const LNB: usize = 32;
        let (client_key, server_key) = keys();
        let a: u64 = 0x1234_5678_9abc_def0u64;
        let b: u64 = 0xfedc_ba98_7654_3210u64;
        let ea = client_key.encrypt_radix(a, LNB);
        let eb = client_key.encrypt_radix(b, LNB);
        let out = server_key.mul_parallelized(&ea, &eb);
        println!("native out blocks = {}", out.blocks().len());
        let dec: u64 = client_key.decrypt_radix(&out);
        println!("native = {dec:#x}, expected_mod = {:#x}", a.wrapping_mul(b));
        assert_eq!(dec, a.wrapping_mul(b));
    }

    #[test]
    fn extended_mul_nb32() {
        const LNB: usize = 32;
        let (client_key, server_key) = keys();
        let a: u64 = 0x1234_5678_9abc_def0u64;
        let b: u64 = 0xfedc_ba98_7654_3210u64;
        let ea = client_key.encrypt_radix(a, LNB);
        let eb = client_key.encrypt_radix(b, LNB);
        let out = mul_wide::<LNB>(&ea, &eb, &server_key);
        println!("mul_wide out blocks = {}", out.blocks().len());
        let dec: U256 = client_key.decrypt_radix(&out);
        let dec = dec.to_low_high_u128().0;
        let expected = (a as u128) * (b as u128);
        println!("mul_wide = {dec:#x}, expected = {expected:#x}");
        assert_eq!(dec, expected);
    }
}

#[cfg(test)]
mod diag3 {
    use super::*;
    use crate::helper::bigint_to_u256;
    use num_bigint::BigInt;
    use tfhe::integer::keycache::IntegerKeyCache;
    use tfhe::integer::{ClientKey, IntegerKeyKind};

    fn keys() -> (ClientKey, ServerKey) {
        IntegerKeyCache::new().get_from_params(
            tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2,
            IntegerKeyKind::Radix,
        )
    }

    #[test]
    fn naive_div_rem_nb_32() {
        const LNB: usize = 32;
        let (client_key, server_key) = keys();
        let p: u128 = 0xFFFF_FFFF_FFFF_FFC5; // 2^64 - 59
        let a: u128 = p - 2;
        let b: u128 = p - 3;
        let ea = client_key.encrypt_radix(a, LNB);
        let eb = client_key.encrypt_radix(b, LNB);
        let out = mul_mod_naive_div_rem::<LNB>(
            &ea,
            &eb,
            bigint_to_u256(&BigInt::from(p)),
            &server_key,
        );
        let dec: u128 = client_key.decrypt_radix(&out);
        assert_eq!(dec, (a * b) % p);
    }
}
