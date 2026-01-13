Purpose: Document the ToyConceptScaffold console experiment and how to reproduce its results.
Persists: None.
Security Risks: None.

# ToyConceptScaffold

A deterministic, console-only toy experiment that contrasts baseline prose prompts with concept scaffold formats under padding noise and paraphrase/reorder stress. It includes two scaffolds: **CE-NEO** (neologism keys) and **CE-ENG** (plain-English keys), letting the paper cite a structure-vs-novelty ablation.

## Quickstart
```bash
dotnet build tools/ToyConceptScaffold/ToyConceptScaffold.csproj

dotnet run --project tools/ToyConceptScaffold -- --mode eval --seed 123 --print-demo
```

## What it demonstrates
- **Baseline vs scaffold robustness**: how explicit key/value scaffolds withstand padding noise compared to prose-like prompts.
- **Structure vs novelty**: CE-NEO vs CE-ENG isolates whether neologisms or structure matter more.
- **Paraphrase/reorder stress**: baseline clauses and scaffold key/value pairs are shuffled or paraphrased to stress order sensitivity.

## Synthetic task
Latent state: 4 binary variables.
- `slip_onset`
- `shear_low`
- `friction_drop`
- `compliance_high`

Ground-truth label and cause:
- If `slip_onset=1` AND `shear_low=1` → Label=SLIP, Cause=SHEAR_LOW
- Else if `compliance_high=1` AND `friction_drop=1` → Label=SQUISH, Cause=FRICTION_DROP
- Else → Label=OK, Cause=NONE

Three prompt formats (same info + CE distractor slots):
1. **BASELINE**: prose-like clauses with synonyms + filler tokens.
2. **CE-NEO**: `slivox=0/1`, `shearvon=0/1`, `frictal=0/1`, `compliq=0/1`, plus `distrax{0-5}=0/1`.
3. **CE-ENG**: `slip_onset=0/1`, `shear_low=0/1`, `friction_drop=0/1`, `compliance_high=0/1`, plus `distractor{0-5}=0/1`.

## Model
- Embedding table `E[V][D]`
- Sum pooling over tokens
- Two linear heads (label and cause)
- Loss = cross-entropy(label) + cross-entropy(cause)
- SGD training

## CLI options
- `--seed 123`
- `--train-samples 30000`
- `--test-samples 5000`
- `--epochs 8`
- `--embed-dim 32`
- `--pad-levels 0,5,15,40,80`
- `--no-paraphrase`
- `--print-demo`
- `--mode eval|train|sanity`

## Sample output (seed=123)
```text
Sample output could not be captured in this environment because the .NET SDK is unavailable.
Run:
dotnet run --project tools/ToyConceptScaffold -- --mode eval --seed 123 --epochs 8 --train-samples 30000 --test-samples 5000 --pad-levels 0,5,15,40,80
dotnet run --project tools/ToyConceptScaffold -- --mode sanity --seed 123 --pad-levels 0,80
...and paste the emitted console output here for publication-quality replication notes.
```

## Notes
- Output is deterministic for a fixed seed (default `123`).
- Training mixes BASELINE/CE-NEO/CE-ENG examples in one model and includes small padding noise to avoid trivial scaffold decoding.
- This is a tiny, interpretable model (not a transformer). It is intentionally minimal to highlight representation effects rather than SOTA performance.
