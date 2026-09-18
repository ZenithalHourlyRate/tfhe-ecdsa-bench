//! GPU benchmark: one CUDA multiplication in the secp256k1 fields.
//!
//! Port of `src/main.rs` to the tfhe-rs 1.8.1 CUDA backend. The CPU benchmark
//! pins all `_parallelized` calls to a 1-thread rayon pool; here all server-side
//! work is issued to the GPU, and each timed operation is followed by a
//! `CudaStreams::synchronize` so the measurement covers the full computation
//! (upload/decrypt are excluded, as on the CPU).
//!
//! * F_p: base field, p = 2^256 - 2^32 - 977 (Pseudo-Mersenne prime)
//! * F_n: scalar field, n = group order (no Mersenne structure -> Barrett)
//!
//! Run with:
//!   cargo run --release --features gpu --bin gpu-bench -- [iters] [fields] [ops] [params]
//!   cargo run --release --features gpu --bin gpu-bench -- selftest
//!
//! `params` is `classic` (default: the same `PARAM_MESSAGE_2_CARRY_2` as the
//! CPU benchmark) or `multibit` (GPU multi-bit PBS, grouping factor 4).

#[path = "../helper.rs"]
mod helper;
#[path = "../gpu_field.rs"]
mod field;

use std::time::{Duration, Instant};

use num_bigint::{BigInt, Sign};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use tfhe::core_crypto::gpu::vec::GpuIndex;
use tfhe::core_crypto::gpu::CudaStreams;
use tfhe::integer::gpu::ciphertext::CudaUnsignedRadixCiphertext;
use tfhe::integer::gpu::CudaServerKey;
use tfhe::integer::{ClientKey, U256, U512};
use tfhe::shortint::parameters::{
    PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128, PARAM_MESSAGE_2_CARRY_2,
};

use crate::field::{mod_barrett, mod_mersenne, mul_mod, mul_mod_naive_div_rem, mul_wide};
use crate::helper::{bigint_to_u256, bigint_to_u512, to_bigint};

/// 128 blocks * 2 message bits = 256 bits.
const NB: usize = 128;

struct Row {
    field: &'static str,
    op: &'static str,
    iters: usize,
    mean_s: f64,
    min_s: f64,
    max_s: f64,
    correct: bool,
}

impl Row {
    fn new(field: &'static str, op: &'static str, times: Vec<Duration>, correct: bool) -> Self {
        let mean_s = times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;
        let min_s = times.iter().map(|d| d.as_secs_f64()).fold(f64::INFINITY, f64::min);
        let max_s = times.iter().map(|d| d.as_secs_f64()).fold(0.0, f64::max);
        Self {
            field,
            op,
            iters: times.len(),
            mean_s,
            min_s,
            max_s,
            correct,
        }
    }
}

fn report(rows: &mut Vec<Row>, row: Row) {
    println!(
        "[{:<3}] {:<30} mean {:>8.3}s  min {:>8.3}s  max {:>8.3}s  correct={}",
        row.field, row.op, row.mean_s, row.min_s, row.max_s, row.correct
    );
    use std::io::Write;
    std::io::stdout().flush().ok();
    rows.push(row);
}

/// Which operations to run (allows trimming the long runs).
#[derive(Clone, Copy)]
struct Ops {
    wide: bool,
    barrett: bool,
    mersenne: bool,
    full: bool,
    naive: bool,
}

impl Ops {
    fn all() -> Self {
        Self {
            wide: true,
            barrett: true,
            mersenne: true,
            full: true,
            naive: true,
        }
    }

    fn parse(s: &str) -> Self {
        let mut ops = Self {
            wide: false,
            barrett: false,
            mersenne: false,
            full: false,
            naive: false,
        };
        for part in s.split(',').map(str::trim) {
            match part {
                "all" => ops = Self::all(),
                "wide" => ops.wide = true,
                "barrett" => ops.barrett = true,
                "mersenne" => ops.mersenne = true,
                "full" => ops.full = true,
                "naive" => ops.naive = true,
                other => panic!("unknown operation filter: {other}"),
            }
        }
        ops
    }
}

/// Time `op` `iters` times, synchronizing the GPU before stopping the clock,
/// and check every output.
fn bench<F, C>(
    streams: &CudaStreams,
    iters: usize,
    label: &str,
    mut op: F,
    check: C,
) -> (Vec<Duration>, bool)
where
    F: FnMut() -> CudaUnsignedRadixCiphertext,
    C: Fn(&CudaUnsignedRadixCiphertext) -> bool,
{
    let mut times = Vec::with_capacity(iters);
    let mut correct = true;
    for i in 0..iters {
        use std::io::Write;
        print!("[start ] {label} (iter {}/{iters})", i + 1);
        std::io::stdout().flush().ok();
        let start = Instant::now();
        let out = op();
        // GPU work is asynchronous: wait for completion before stopping the clock.
        streams.synchronize();
        let elapsed = start.elapsed();
        println!(" -> {:.3}s", elapsed.as_secs_f64());
        times.push(elapsed);
        correct &= check(&out);
    }
    (times, correct)
}

fn random_element(modulus: &BigInt, rng: &mut StdRng) -> BigInt {
    loop {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        let x = BigInt::from_bytes_le(Sign::Plus, &bytes) % modulus;
        if x != BigInt::from(0) {
            return x;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_field(
    streams: &CudaStreams,
    client_key: &ClientKey,
    server_key: &CudaServerKey,
    field_name: &'static str,
    modulus_big: &BigInt,
    modulus: U256,
    a_clear: &BigInt,
    b_clear: &BigInt,
    a_enc: &CudaUnsignedRadixCiphertext,
    b_enc: &CudaUnsignedRadixCiphertext,
    iters: usize,
    ops: Ops,
    rows: &mut Vec<Row>,
) {
    let expected_prod: BigInt = a_clear * b_clear;
    let expected_mod = &expected_prod % modulus_big;
    let expected_umod: U256 = bigint_to_u256(&expected_mod);
    let expected_u512: U512 = bigint_to_u512(&expected_prod);

    let check_mod = |out: &CudaUnsignedRadixCiphertext| -> bool {
        let dec: U256 = client_key.decrypt_radix(&out.to_radix_ciphertext(streams));
        if dec != expected_umod {
            eprintln!(
                "  [check] {field_name} mod mismatch: got {}, expected {}",
                to_bigint(dec),
                expected_mod
            );
            false
        } else {
            true
        }
    };
    let check_prod = |out: &CudaUnsignedRadixCiphertext| -> bool {
        let dec: U512 = client_key.decrypt_radix(&out.to_radix_ciphertext(streams));
        if dec != expected_u512 {
            eprintln!(
                "  [check] {field_name} product mismatch: got {}, expected {}",
                to_bigint(dec),
                expected_prod
            );
            false
        } else {
            true
        }
    };

    if ops.full {
        let label = format!("{field_name} FULL mul_mod");
        let (times, correct) = bench(
            streams,
            iters,
            &label,
            || mul_mod::<NB>(a_enc, b_enc, modulus, server_key, streams),
            &check_mod,
        );
        report(rows, Row::new(field_name, "FULL mul_mod", times, correct));
    }

    // 1. raw wide multiplication: 256-bit x 256-bit -> exact 512-bit product
    if ops.wide {
        let label = format!("{field_name} wide mul");
        let (times, correct) = bench(
            streams,
            iters,
            &label,
            || mul_wide::<NB>(a_enc, b_enc, server_key, streams),
            &check_prod,
        );
        report(
            rows,
            Row::new(field_name, "wide mul (256x256 -> 512)", times, correct),
        );
    }

    // product is reused for the reduction-only measurements
    let product = if ops.barrett || ops.mersenne {
        Some(mul_wide::<NB>(a_enc, b_enc, server_key, streams))
    } else {
        None
    };
    streams.synchronize();

    // 2. Barrett reduction only (this is what F_n uses)
    if ops.barrett {
        let label = format!("{field_name} reduce Barrett");
        let (times, correct) = bench(
            streams,
            iters,
            &label,
            || mod_barrett::<NB>(product.as_ref().unwrap(), modulus, server_key, streams),
            &check_mod,
        );
        report(
            rows,
            Row::new(field_name, "reduce only (Barrett)", times, correct),
        );
    }

    // 3. Mersenne reduction only (dispatches to Barrett for F_n)
    if ops.mersenne {
        let label = format!("{field_name} reduce Mersenne");
        let (times, correct) = bench(
            streams,
            iters,
            &label,
            || mod_mersenne::<NB>(product.as_ref().unwrap(), modulus, server_key, streams),
            &check_mod,
        );
        report(
            rows,
            Row::new(field_name, "reduce only (Mersenne|Barrett)", times, correct),
        );
    }

    // 5. naive baseline: wide mul + arithmetic scalar div/rem
    if ops.naive {
        let label = format!("{field_name} mul + scalar div_rem");
        let (times, correct) = bench(
            streams,
            iters,
            &label,
            || mul_mod_naive_div_rem::<NB>(a_enc, b_enc, modulus, server_key, streams),
            &check_mod,
        );
        report(
            rows,
            Row::new(field_name, "mul + scalar div_rem", times, correct),
        );
    }
}

/// Small-NB (16-bit) validation of the GPU port: catches porting mistakes on
/// cheap ciphertexts, before committing to the 256-bit benchmark.
fn selftest(streams: &CudaStreams) {
    const TNB: usize = 8; // 8 blocks * 2 message bits = 16 bits

    println!("GPU self-test: NB={TNB} (16-bit values), PARAM_MESSAGE_2_CARRY_2");
    let client_key = ClientKey::new(PARAM_MESSAGE_2_CARRY_2);
    let server_key = CudaServerKey::new(&client_key, streams);
    streams.synchronize();

    let upload = |v: u64| -> CudaUnsignedRadixCiphertext {
        CudaUnsignedRadixCiphertext::from_radix_ciphertext(
            &client_key.encrypt_radix(v, TNB),
            streams,
        )
    };
    let decrypt = |ct: &CudaUnsignedRadixCiphertext| -> u64 {
        client_key.decrypt_radix(&ct.to_radix_ciphertext(streams))
    };

    // exact wide product (16 x 16 -> 32 bits)
    let (a, b) = (0xFFFFu64, 0x1234u64);
    let prod = mul_wide::<TNB>(&upload(a), &upload(b), &server_key, streams);
    assert_eq!(decrypt(&prod), a * b, "wide mul");

    // 251 = 2^8 - 5 -> Pseudo-Mersenne path; 263 -> Barrett path
    for &(p, a, b) in &[
        (251u64, 250u64, 249u64),
        (251, 123, 200),
        (263, 262, 261),
        (263, 123, 200),
    ] {
        let out = mul_mod::<TNB>(
            &upload(a),
            &upload(b),
            bigint_to_u256(&BigInt::from(p)),
            &server_key,
            streams,
        );
        assert_eq!(decrypt(&out), (a * b) % p, "mul_mod p={p} a={a} b={b}");
    }

    // naive division baseline
    let (p, a, b) = (263u64, 262u64, 261u64);
    let out = mul_mod_naive_div_rem::<TNB>(
        &upload(a),
        &upload(b),
        bigint_to_u256(&BigInt::from(p)),
        &server_key,
        streams,
    );
    assert_eq!(decrypt(&out), (a * b) % p, "naive div_rem");

    println!("GPU self-test passed");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(String::as_str) == Some("selftest") {
        let streams = CudaStreams::new_single_gpu(GpuIndex::new(0));
        selftest(&streams);
        return;
    }

    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let fields_arg = args.get(2).map(String::as_str).unwrap_or("p,n");
    let ops = args.get(3).map(|s| Ops::parse(s)).unwrap_or_else(Ops::all);
    let params_name = args.get(4).map(String::as_str).unwrap_or("classic");
    let run_p = fields_arg.split(',').any(|f| f.trim() == "p");
    let run_n = fields_arg.split(',').any(|f| f.trim() == "n");

    println!("tfhe-rs       : 1.8.1 (CUDA backend)");
    println!("iterations    : {} per operation", iters);
    println!("fields        : p={} n={}", run_p, run_n);

    let streams = CudaStreams::new_single_gpu(GpuIndex::new(0));

    let t0 = Instant::now();
    let (client_key, server_key) = match params_name {
        "multibit" => {
            let ck = ClientKey::new(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128);
            let sk = CudaServerKey::new(&ck, &streams);
            (ck, sk)
        }
        "classic" => {
            let ck = ClientKey::new(PARAM_MESSAGE_2_CARRY_2);
            let sk = CudaServerKey::new(&ck, &streams);
            (ck, sk)
        }
        other => panic!("unknown parameters: {other} (use classic|multibit)"),
    };
    streams.synchronize();
    println!(
        "parameters    : {} (128 blocks = 256 bits)",
        if params_name == "multibit" {
            "PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128"
        } else {
            "PARAM_MESSAGE_2_CARRY_2"
        }
    );
    println!("key setup     : {:.1?} (not part of the benchmark)", t0.elapsed());

    // secp256k1 field constants
    let p_big = BigInt::parse_bytes(
        b"115792089237316195423570985008687907853269984665640564039457584007908834671663",
        10,
    )
    .unwrap();
    let n_big = BigInt::parse_bytes(
        b"115792089237316195423570985008687907852837564279074904382605163141518161494337",
        10,
    )
    .unwrap();
    let p: U256 = bigint_to_u256(&p_big);
    let n: U256 = bigint_to_u256(&n_big);

    let mut rng = StdRng::seed_from_u64(0xEC05A_B0A7);
    let a_p = random_element(&p_big, &mut rng);
    let b_p = random_element(&p_big, &mut rng);
    let a_n = random_element(&n_big, &mut rng);
    let b_n = random_element(&n_big, &mut rng);

    // Fresh ciphertexts are created (and uploaded) before the benchmark starts,
    // exactly like the CPU version.
    let upload = |ct| CudaUnsignedRadixCiphertext::from_radix_ciphertext(&ct, &streams);
    let a_p_enc = upload(client_key.encrypt_radix(bigint_to_u256(&a_p), NB));
    let b_p_enc = upload(client_key.encrypt_radix(bigint_to_u256(&b_p), NB));
    let a_n_enc = upload(client_key.encrypt_radix(bigint_to_u256(&a_n), NB));
    let b_n_enc = upload(client_key.encrypt_radix(bigint_to_u256(&b_n), NB));
    streams.synchronize();

    let mut rows = Vec::new();
    if run_p {
        run_field(
            &streams, &client_key, &server_key, "F_p", &p_big, p, &a_p, &b_p, &a_p_enc, &b_p_enc,
            iters, ops, &mut rows,
        );
    }
    if run_n {
        run_field(
            &streams, &client_key, &server_key, "F_n", &n_big, n, &a_n, &b_n, &a_n_enc, &b_n_enc,
            iters, ops, &mut rows,
        );
    }

    println!();
    println!(
        "| {:<4} | {:<30} | {:>5} | {:>12} | {:>12} | {:>12} | {:>7} |",
        "field", "operation", "iters", "mean (s)", "min (s)", "max (s)", "correct"
    );
    println!(
        "|{:-<6}|{:-<32}|{:-<7}|{:-<14}|{:-<14}|{:-<14}|{:-<9}|",
        "", "", "", "", "", "", ""
    );
    for row in &rows {
        println!(
            "| {:<4} | {:<30} | {:>5} | {:>12.3} | {:>12.3} | {:>12.3} | {:>7} |",
            row.field,
            row.op,
            row.iters,
            row.mean_s,
            row.min_s,
            row.max_s,
            if row.correct { "yes" } else { "NO" }
        );
    }

    // Summarise the comparison requested: one multiplication in F_p vs F_n.
    let find = |field: &str, op: &str| -> Option<f64> {
        rows.iter()
            .find(|r| r.field == field && r.op == op)
            .map(|r| r.mean_s)
    };
    if let (Some(fp), Some(fn_)) = (find("F_p", "FULL mul_mod"), find("F_n", "FULL mul_mod")) {
        println!();
        println!(
            "FULL mul_mod mean: F_p = {:.3} s, F_n = {:.3} s, ratio F_n/F_p = {:.2}x",
            fp,
            fn_,
            fn_ / fp
        );
    }
    let bad = rows.iter().filter(|r| !r.correct).count();
    println!();
    if bad == 0 {
        println!("All results decrypted to the expected value.");
    } else {
        println!("WARNING: {} benchmark rows produced wrong results!", bad);
    }
}
