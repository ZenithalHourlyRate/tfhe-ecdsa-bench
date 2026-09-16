# tfhe-ecdsa-bench

Single-threaded benchmark of **one modular multiplication** in the two
secp256k1 prime fields, using the latest `tfhe-rs`:

* `F_p`: base field, `p = 2^256 - 2^32 - 977` (Pseudo-Mersenne prime)
* `F_n`: scalar field, `n` = secp256k1 group order (no useful Mersenne form)

## Does latest tfhe-rs support multiplication in these fields?

**No.** As of `tfhe-rs 1.8.1` (the latest crates.io release) there is no
finite-field / modular-multiplication API for ciphertexts. What the crate
provides are the integer primitives needed to build one:

* `ServerKey::mul_assign_parallelized` (schoolbook radix multiplication)
* `extend_radix_with_trivial_zero_blocks_msb`, `trim_radix_blocks_msb_assign`
* `scalar_mul_parallelized`, `scalar_left_shift_parallelized`,
  `scalar_right_shift_parallelized`
* `scalar_ge_parallelized`, `scalar_div_rem_parallelized`, etc.

A search of the `tfhe-rs 1.8.1` sources and of the current `main` branch for
`mul_mod` / `modular` only finds the *native* (`u32`/`u64`) helpers inside the
internal NTT implementation (`tfhe-ntt`), not anything operating on
`RadixCiphertext`.

Consequently the benchmark implements field multiplication with the algorithms
from the original [TFHECDSA](https://github.com/Tetration-Lab/TFHECDSA)
repository, ported from `tfhe 0.3.0` to the `tfhe-rs 1.8.1` API:

* `mul_wide`: zero-extend `a` to `2*NB` blocks, multiply by `b`, keep the exact
  512-bit product (two 256-bit operands).
* `mod_mersenne`: two passes of Pseudo-Mersenne reduction
  (`x = a*2^n + b  =>  x == c*a + b mod p`) plus a final conditional subtract.
  Used for `F_p`.
* `mod_barrett`: `q = (x * floor(2^512 / m)) >> 512`, `x - q*m`, then a final
  conditional subtract. Used for `F_n` (its `c = 2^256 - n` is too large for the
  Mersenne trick) and as a comparison point for `F_p`.

API changes handled during the port from 0.3.0 -> 1.8.1:

* comparators now return `BooleanBlock` instead of `RadixCiphertext`
  (`into_radix(NB, &sks)` is used as the selector);
* `IntegerKeyCache::get_from_params` now takes an `IntegerKeyKind`;
* `add`/`sub` now **require both operands to have the same number of
  blocks** (`assert_eq!` in `add.rs`/`sub.rs`), so the shorter operand is
  zero-extended before those calls;
* `mul_assign_parallelized` only computes a correct low part when
  `rhs.blocks() >= lhs.blocks()`: `compute_terms_for_mul_low` deliberately
  drops the carry of the most significant `rhs` block, which lands *inside*
  the result when `rhs` is narrower. The original repo relied on the old
  (asymmetric) behaviour. Here `mul_wide` zero-extends **both** operands to
  `2*NB` blocks before multiplying, which is the correct usage on 1.8.1 and
  yields the exact `2*b`-bit product.

## Benchmark setup

* `tfhe-rs 1.8.1`, features `integer`, `internal-keycache` (default `avx512`).
* `PARAM_MESSAGE_2_CARRY_2`, `NB = 128` blocks = 256 message bits.
* All server-side work runs inside a `rayon` pool with **1 thread**
  (`ThreadPoolBuilder::num_threads(1)` + `pool.install`), so every
  `_parallelized` tfhe-rs call is strictly single-threaded.
* Inputs: deterministic random `a, b` reduced into `[1, m)`; fresh ciphertexts.
* Every benchmarked operation's output is decrypted and checked against a
  `num_bigint` reference; `correct = yes` means the result matched.

Measured operations per field:

| operation | meaning |
|---|---|
| `wide mul (256x256 -> 512)` | `a * b` only (no reduction) |
| `reduce only (Barrett)` | `mod_barrett` on a precomputed product |
| `reduce only (Mersenne\|Barrett)` | `mod_mersenne` on a precomputed product (dispatches to Barrett for `F_n`) |
| `FULL mul_mod` | `a * b mod m` (wide mul + reduction) |
| `mul + scalar div_rem` | naive baseline: wide mul + arithmetic `scalar_div_rem` |

## Run

```bash
cargo run --release -- [iterations]     # default: 3 iterations per op
```

The first run generates and caches the keys (`~/.cache/tfhe-rs`); subsequent
runs reuse them from disk.

## Results

See `RESULTS.md` (filled in from an actual run).
