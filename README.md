# NeuroGolf 2026 — Minimal ONNX Networks for ARC-AGI

Competition entry for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) (Kaggle, featured at IJCAI-ECAI 2026): build the **smallest possible neural network** that exactly implements each of the 400 ARC-AGI-1 grid transformations.

**Result: public LB 7090.11 (top ~37% of 2,754 teams) reached within ~24h of tooling work, with a locally-calibrated scorer accurate to the decimal against the official judge.**

## The game

Each task ships a set of `input → output` grid examples (train + test + [ARC-GEN](https://github.com/google/ARC-GEN) pairs, plus a private benchmark). A submission is one ONNX network per task, mapping a `[1,10,30,30]` one-hot color tensor to the expected output tensor, judged by exact match of `(output > 0)`. Score per task:

```
max(1, 25 − ln(params + intermediate_tensor_bytes))
```

Correct-but-tiny beats trained-and-approximate: a zero-cost `Transpose` graph scores a full 25, a gradient-trained conv scores ~16. So networks are **compiled, not trained** — weights are constructed analytically from the transformation's exact semantics.

## What's here

| Path | Purpose |
|---|---|
| `src/ng/builder.py` | Compact ONNX graph-construction DSL with cost accounting |
| `src/ng/check.py` | Judge-faithful validator + **robustness harness**: validates against 300+ freshly-generated pairs from the official generator (a proxy for the private benchmark — calibration verified: predicted 7089.6, actual LB 7090.11) |
| `src/ng/show.py` | Task inspector: grids, size relations, readable reference spec |
| `src/ng/macros.py` | Reusable tensor idioms: masked flood-fill, raster-cascade fixpoints, exact template matching |
| `src/ng/merge_bundles.py` | Best-per-task merge across candidate sources, all re-verified robustly |
| `src/ng/lossless_pass.py` | Behavior-preserving graph optimization sweep (verified, not assumed) |
| `src/ng/package.py` | Submission packaging + Kaggle upload |
| `src/ng/tasks/tNNN.py` | Per-task network builders (hand-compiled semantics) |
| `solutions/` | Current best networks + audit reports (merge, lossless, risk ledger) |
| `AGENT_GUIDE.md`, `FLEET_NOTES.md` | Playbooks used to parallelize task-solving across LLM agents |
| `SETUP.md` | Full data reconstruction (the `data/` tree is excluded, ~870MB) |

## Key findings

1. **The generator is the ground truth.** Task semantics are defined by Google's ARC-GEN generator code, not by the "concept" of the original ARC task. Example: hole-filling in task 002 is a *raster-order in-place scan* (`is_surrounded`), not a flood fill — locally indistinguishable in most examples, divergent on ~3% of generated pairs. Some tasks contain provably irreducible ambiguity (identical input patterns, different labels depending on hidden generator state); these are tracked in a risk ledger with measured failure rates.
2. **Fresh-pair fuzzing predicts the private benchmark.** Generating new pairs with the official generator and requiring 100% correctness made local scores match the hidden leaderboard to the decimal, across two submissions.
3. **Cost anatomy.** Node attributes are free; tensors named `input`/`output` are free; below ~17 points/task the cost is dominated by intermediate tensor bytes, not parameters — so the craft is minimizing the number and dtype-size of intermediates (bool = 1B/elem, fp16 convs verified working on ORT CPU), fusing thresholds into conv biases, and replacing dense weights with `Gather` indices.
4. **Numerical safety in unrolled iteration.** Loop-free propagation (flood fills, cascades) needs kernels ≤0.25 against float32 overflow and periodic `Sign` renormalization against denormal flush — both bugs manifest only on long dependency chains, i.e., exactly on the private-style pairs.

## Attribution

Built on and merging openly shared community work, per the competition's open-sharing rules:

- [google/ARC-GEN](https://github.com/google/ARC-GEN) (Apache-2.0) — official benchmark generator, used as semantic ground truth and fuzzing oracle
- [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) — original task corpus
- [farukalamai/neurogolf-2026-toolkit](https://github.com/farukalamai/neurogolf-2026-toolkit) (Apache-2.0) — forked in `toolkit/`: calibrated validator port and baseline strategies
- Public Kaggle notebook bundles by @kojimar, @seddiktrk, @boristown, @octaviograu, @konbu17, @vyankteshdwivedi, @afr1ste, @franksunp, @mirzayasirabdullah07, @needless090 — candidate networks merged (and re-verified) per task
- ONNX compilation cookbook distilled from community write-ups (see `data/ref_Neurogolf/TECHNIQUES.md` after setup); reference Python solutions from [xsot/google-code-golf-2025](https://github.com/xsot/google-code-golf-2025) (Apache-2.0) and [michaelhodel/re-arc](https://github.com/michaelhodel/re-arc)

License: Apache-2.0.
