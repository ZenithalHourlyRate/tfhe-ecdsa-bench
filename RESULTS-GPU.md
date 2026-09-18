# Benchmark results — one field multiplication on GPU

Port of `src/main.rs` to the tfhe-rs 1.8.1 CUDA backend
(`src/bin/gpu_bench.rs` + `src/gpu_field.rs`). Same algorithms, same
`NB = 128` blocks (2-bit message / 2-bit carry = 256 bits), same deterministic
inputs. Every row is decrypted with the CPU `ClientKey` and checked against
`num_bigint` (`correct = yes` everywhere).

Environment:

* `tfhe-rs 1.8.1` (crates.io) with the `gpu` feature, `tfhe-cuda-backend 0.16.0`
* NVIDIA RTX PRO 6000 Blackwell, 96 GB, driver 590.44.01, CUDA 13.0
* all server-side work issued on the GPU; each timed operation is followed by
  `CudaStreams::synchronize()` so the clock covers the full computation
* ciphertext upload and result decryption are excluded from the timings
* key setup ≈ 0.5 s (not part of the benchmark)
* the NB=8 GPU self-test (`gpu-bench selftest`) passes

Run with:

```bash
cargo run --release --features gpu --bin gpu-bench -- 1 p,n full,wide,barrett,mersenne classic
cargo run --release --features gpu --bin gpu-bench -- 1 p,n full,wide,barrett,mersenne multibit
```

## Headline: one multiplication in F_p and F_n

| field | operation | classic PBS | multi-bit PBS (group 4) | correct |
|---|---|---:|---:|:---:|
| **F_p** (`a*b mod p`) | FULL `mul_mod` | **37.395 s** | **16.021 s** | yes |
| **F_n** (`a*b mod n`) | FULL `mul_mod` | **48.301 s** | **21.142 s** | yes |

* `F_n / F_p` mean-time ratio: **1.29x** (classic), **1.32x** (multi-bit)
* secp256k1 base field: `p = 2^256 - 2^32 - 977` (Pseudo-Mersenne)
* secp256k1 scalar field: `n` = group order (no usable Mersenne structure)
* the difference is entirely the reduction algorithm:
  Pseudo-Mersenne for `F_p` vs Barrett for `F_n`

## Clean breakdown (one iteration each)

| operation | F_p classic | F_n classic | F_p multi-bit | F_n multi-bit |
|---|---:|---:|---:|---:|
| wide mul (256x256 -> 512) | 32.524 s | 31.923 s | 13.989 s | 13.998 s |
| reduce only (Barrett) | 16.203 s | 16.188 s | 6.869 s | 6.930 s |
| reduce only (Mersenne, dispatches to Barrett for F_n) | 4.965 s | 16.181 s | 1.895 s | 6.850 s |
| FULL `mul_mod` | 37.395 s | 48.301 s | 16.021 s | 21.142 s |
| naive `mul + scalar div_rem` | not run | not run | not run | not run |

Consistency checks:

* classic `F_p`: `wide + Mersenne = 32.524 + 4.965 = 37.489 s` vs
  FULL `37.395 s` (+0.25%).
* classic `F_n`: `wide + Barrett = 31.923 + 16.188 = 48.111 s` vs
  FULL `48.301 s` (-0.39%).
* multi-bit `F_p`: `13.989 + 1.895 = 15.884 s` vs FULL `16.021 s` (-0.86%).
* multi-bit `F_n`: `13.998 + 6.930 = 20.928 s` vs FULL `21.142 s` (-1.01%).
* Pseudo-Mersenne is **3.26x** (classic) / **3.63x** (multi-bit) faster than
  Barrett on the same 512-bit product.
* The exact ciphertext-ciphertext multiplication dominates: 87% of the classic
  `F_p` and 66% of the classic `F_n` field multiplication (multi-bit: 87%
  and 66% as well).

## GPU vs single-threaded CPU (`RESULTS.md`)

| operation | CPU (1 thread) | GPU classic | GPU multi-bit | GPU classic speedup | GPU multi-bit speedup |
|---|---:|---:|---:|---:|---:|
| `F_p` FULL `mul_mod` | 569.475 s | 37.395 s | 16.021 s | **15.2x** | **35.5x** |
| `F_n` FULL `mul_mod` | 840.389 s | 48.301 s | 21.142 s | **17.4x** | **39.7x** |
| wide mul | 523.691 s | 32.524 s | 13.989 s | 16.1x | 37.4x |
| Barrett reduce | 275.642 s | 16.203 s | 6.869 s | 17.0x | 40.1x |
| Mersenne reduce (F_p) | 44.936 s | 4.965 s | 1.895 s | 9.0x | 23.7x |

The classic parameter set is exactly the CPU benchmark's
`PARAM_MESSAGE_2_CARRY_2` (= `..._KS_PBS_TUNIFORM_2M128`), so those two runs are
directly comparable. The multi-bit set
(`PARAM_GPU_MULTI_BIT_GROUP_4_..._TUNIFORM_2M128`, grouping factor 4) is the
faster GPU setting and is the one to use in practice.

### What one multiplication costs (GPU)

* `F_p`: ~37 s classic PBS, ~16 s multi-bit PBS (`a * b mod p`, 256-bit
  operands encrypted in 128 two-bit blocks)
* `F_n`: ~48 s classic PBS, ~21 s multi-bit PBS

## Naive baseline (generic division) — not run

As on the CPU side, the `mul + scalar_div_rem` path was not pursued: it was
started and aborted, the specialized reductions are clearly the right approach.
The benchmark supports the `naive` op key if it is ever wanted.

## GPU porting notes

The GPU algorithms are direct ports of `field.rs`; the differences handled
here are backend constraints, not mathematical changes:

1. `add`/`sub`/`mul` require both operands to have the same number of blocks,
   as on the CPU.
2. The CUDA scalar comparison (`scalar_ge`, used by `modulo_fast`) requires an
   **even** number of radix blocks (or exactly one). `modulo_fast` receives
   `NB` or `NB + 1` blocks, so odd widths are extended with a single trivial
   zero MSB block before the comparison.
3. The GPU backend has no `BooleanBlock::into_radix` equivalent: the
   comparison result (0/1) is zero-extended to `NB` blocks and multiplied by
   the constant, yielding 0 or the constant.
4. Block counts cannot be queried from `CudaLweCiphertextList` (crate-private);
   the port reads the count from `CudaRadixCiphertext::info.blocks`.
5. All GPU operations are non-mutating and take a `&CudaStreams`; `mul` returns
   the low `lhs.blocks()` blocks of the product, so `mul_wide` keeps the same
   zero-extend-both-operands-to-`2*NB` strategy as the CPU port.
