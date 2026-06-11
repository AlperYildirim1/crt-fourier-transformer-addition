# Pythia Addition: Residue and Magnitude Channels

Causal experiments on how Pythia-6.9B computes two digit addition (`a+b` for `a, b` in `[0, 99]`).

This builds directly on:

> **Language Models Use Trigonometry to Do Addition**
> Subhash Kantamneni, Max Tegmark (2025). [arXiv:2502.00873](https://arxiv.org/abs/2502.00873)

## What we take from the paper

The paper shows that LLMs (GPT-J, Pythia-6.9B, Llama3.1-8B) represent numbers as a generalized helix: a linear component plus Fourier circles with periods T = [2, 5, 10, 100]. It proposes that addition works via the "Clock" algorithm: the model builds helix(a+b) from helix(a) and helix(b), then reads the answer out to logits. Evidence comes from activation patching with fitted helices, ablations of the helix subspace, and component level analysis of GPT-J.

We reuse:

- The same task and prompt format ("Output ONLY a number. {a}+{b}=")
- The helix basis (cos/sin at T = 2, 5, 10, 100 plus a linear term), fit with ridge regression on residual stream activations
- Pythia-6.9B as the model

## What we add

The paper treats the helix mostly as one object. We split it apart and test each piece causally, all on Pythia-6.9B.

**1. Two channels inside the helix (Tests 1, 2, 2B).**
Persistent ablation of the T2/T5/T10 subspace destroys the units digit of the answer (same units on wrong answers: 5%). Ablation of T100/LINEAR does the opposite: the units digit survives (same units on wrong answers: 86-93%) but the decade is destroyed, and errors become multiples of 10. So the helix factors into a low period residue channel and a magnitude channel. Matched rank random ablations barely hurt the model (94% exact), so this is not generic rank damage.

**2. Wrong frequency controls (Tests 3, 4).**
Ablating fake periods T3/T7 also hurts the model, which would be an objection to subspace specificity. We show most of that damage is overlap: after orthogonalizing T3/T7 against the real helix span, accuracy recovers substantially (T7: 55% to 91% exact). Recovery is large but not total for T3, which we report as is.

**3. Residue steering with built in nulls (Test 6).**
Instead of patching in fits of the true answer, we pin the residue subspace to counterfactual residues for sum + delta. With delta = 1, the output follows the injected residue 96% of the time (follow10 vs stay10). delta = 10 is a designed null (s and s+10 share all low period residues) and behaves as a null. delta = 2 leaves parity untouched and delta = 5 leaves mod5 untouched, exactly as predicted. Caveat: steering a single period alone (T5 only or T10 only) breaks the residue but does not redirect it, so the readout seems to need cross period consistency.

**4. Magnitude steering (Test 7).**
T100/LINEAR steering moves the decade coarsely while keeping the units digit high, but it is noisy (delta = 20 works, delta = 10 mostly does not). We treat this as supporting evidence only.

**5. Source token residue arithmetic (Test 8, main new result).**
Everything above is about the answer representation. Here we inject independent counterfactual residues on the a token and b token, and ask whether the output follows the modular sum of what we injected. In the pure condition (independent mod2 and mod5 injections), the output units digit follows the CRT composed target 42% of the time vs 17% staying at the true answer (chance 10%). This is direct causal evidence that source residues are composed by modular addition inside the model, which the paper hypothesized but could not isolate.

**6. Natural errors match the channels (Test 0).**
The model's natural wrong answers preserve mod2 at 99.9% and mod10 at 81%, and are dominated by plus or minus 10 and 20 shifts. So natural failures are magnitude channel failures with an intact residue channel, which is what the channel decomposition predicts.

## Results at a glance

| Test | Question | Result |
|---|---|---|
| 0 | What do natural errors look like? | Mostly +-10/+-20, units digit preserved |
| 1, 2 | Is the helix one thing? | No: residue channel (T2/T5/T10) vs magnitude channel (T100/LINEAR) |
| 3, 4 | Is T3/T7 damage an objection? | Mostly overlap with the real span |
| 5 | Which period hits which residue? | Selectivity matrix, periods fingerprint cleanly |
| 6 | Can we steer residues? | Yes, 96% follow at delta=1, nulls behave |
| 7 | Can we steer magnitude? | Coarsely, noisy |
| 8 | Are source residues actually added? | Yes, injected mod2 x mod5 composes into output mod10 |

## Running

Single notebook style pipeline. Requires a GPU with enough memory for Pythia-6.9B in fp16 (about 14 GB plus headroom).

```
pip install torch transformers scikit-learn pandas tqdm
```

Run the cells in order: setup, baseline filter, then Tests 0 through 8. Each test saves a CSV to `OUT_DIR` and the dashboard cells at the end print human readable tables plus a `human_summary.md`.

Set `MAX_EVAL_EXAMPLES = 1000` for a smoke test; `None` runs the full 7951 baseline correct examples.

## Limitations

- One model (Pythia-6.9B), one prompt format, but preliminary GPT-J tests are clearer and shows the same story. Will be added.
- T3 wrong frequency recovery is partial, not complete
- Single period steering fails, so the channels are not independently steerable
- Magnitude steering is weak relative to magnitude ablation
- GPT-J parity runs not done yet

## Citation

If you use this, please cite the original paper:

```
@article{kantamneni2025trigonometry,
  title={Language Models Use Trigonometry to Do Addition},
  author={Kantamneni, Subhash and Tegmark, Max},
  journal={arXiv preprint arXiv:2502.00873},
  year={2025}
}
```
