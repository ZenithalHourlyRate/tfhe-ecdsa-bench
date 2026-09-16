# Benchmark results — one field multiplication, single-threaded CPU

Environment:

* `tfhe-rs 1.8.1` (crates.io, default `avx512` features), `integer` + `internal-keycache`
* AMD Ryzen AI MAX+ 395 (Zen 5), 32 hardware threads, 124 GB RAM
* `PARAM_MESSAGE_2_CARRY_2` (TUniform 2M128), `NB = 128` blocks = 256 bits
* all tfhe-rs server operations executed in a rayon pool with **1 thread**
  (`rayon::ThreadPoolBuilder::num_threads(1)` + `pool.install`)
* inputs: deterministic random `a, b` in `[1, m)`, fresh ciphertexts
* every output decrypted and checked against `num_bigint` (`correct = yes`)

Run with:

```bash
cargo run --release -- <iters> <fields> <ops>
# all operations, 1 iteration, both fields:
cargo run --release -- 1 p,n all
```

## Headline: one multiplication in F_p and F_n

| field | operation | run 1 | run 2 | mean | spread | correct |
|---|---|---:|---:|---:|---:|:---:|
| **F_p** (`a*b mod p`) | FULL `mul_mod` | 567.741 s | 571.208 s | **569.475 s** (9m29s) | 0.6% | yes |
| **F_n** (`a*b mod n`) | FULL `mul_mod` | 842.352 s | 838.426 s | **840.389 s** (14m00s) | 0.5% | yes |

* `F_n / F_p` mean-time ratio: **1.48x**
* secp256k1 base field: `p = 2^256 - 2^32 - 977` (Pseudo-Mersenne)
* secp256k1 scalar field: `n` = group order (no usable Mersenne structure)
* the difference is entirely the reduction algorithm:
  Pseudo-Mersenne for `F_p` vs Barrett for `F_n`.

## Clean breakdown (single iteration each)

| operation | F_p | F_n |
|---|---:|---:|
| wide mul (256x256 -> 512) | 523.691 s | 528.274 s |
| reduce only (Barrett) | 275.642 s | 314.279 s |
| reduce only (Mersenne, dispatches to Barrett for F_n) | 44.936 s | 314.211 s |
| FULL `mul_mod` | 567.741 s | 842.352 s |
| naive `mul + scalar div_rem` | see note | see note |

Consistency checks:

* `F_p`: `wide + Mersenne = 523.691 + 44.936 = 568.627 s` vs
  FULL `567.741 s` (−0.16%).
* `F_n`: `wide + Barrett = 528.274 + 314.279 = 842.553 s` vs
  FULL `842.352 s` (−0.02%).
* Pseudo-Mersenne is **6.13x faster** than Barrett on the same 512-bit
  product (`275.642 s` vs `44.936 s` for `F_p`).
* The exact ciphertext-ciphertext multiplication dominates: 92% of the `F_p`
  and 63% of the `F_n` field multiplication.

### What one multiplication costs (single thread)

* `F_p`: ~9.5 minutes (`a * b mod p`, 256-bit operands encrypted in 128
  two-bit blocks).
* `F_n`: ~14 minutes (`a * b mod n`, same layout).

For scale: the tutorial's full ECDSA signing needs thousands of field
multiplications plus inversions and scalar multiplications, which is why it
takes days on one thread — one field multiplication alone is minutes.

## Naive baseline (generic division) — not pursued

The naive route (`mul_wide` then `ServerKey::scalar_div_rem_parallelized`) is
not directly available in 1.8.1 for a 256-bit divisor on a 512-bit numerator:
the API asserts `scalar_bits >= encrypted_bits`, so the divisor has to be
widened to `U512`. A run of that path was started for `F_p` and was still
running after the `mul_wide` part (~9 min) plus >10 min of division, so it was
abandoned: the specialized reductions are clearly the right approach. The
benchmark supports the `naive` op key if it is ever wanted.

## Notes on the port to tfhe-rs 1.8.1

The old repo targeted `tfhe 0.3.0`; the following 1.8.1 API differences were
found and handled:

1. comparators return `BooleanBlock`, not `RadixCiphertext`
   (`selector_zero_constant` now uses `BooleanBlock::into_radix`);
2. `add`/`sub` panic unless both operands have the same number of blocks
   (`add.rs:560`), so the narrower operand is zero-extended;
3. `IntegerKeyCache::get_from_params` takes an `IntegerKeyKind`;
4. `mul_assign_parallelized` is only correct when `rhs.blocks() >= lhs.blocks()`
   — `compute_terms_for_mul_low` intentionally drops the carry of the most
   significant `rhs` block, which lands *inside* the result when `rhs` is
   narrower. `mul_wide` therefore zero-extends **both** operands to `2*NB`
   before multiplying. (The old repo's `extend(a, NB); mul(a, b)` pattern
   silently produced wrong high bits on 1.8.1; this was caught by the
   decryption checks in this benchmark.)
5. `scalar_div_rem_parallelized` requires the scalar type to cover all
   encrypted bits.

The port is validated by 13 in-tree tests (small-NB, fast) covering Mersenne
and Barrett reduction, exactness of the wide product at NB=8/32/64, the
Barrett/Mersenne agreement, and the large-value conversions.
