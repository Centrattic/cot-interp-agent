# Qwen3-32B SAE Feature Labels

Natural-language labels for 19,970 features of the 65k-wide BatchTopK SAE
at layer 32 (50% depth) of [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).

## SAE

- **Source**: [adamkarvonen/qwen3-32b-saes](https://huggingface.co/adamkarvonen/qwen3-32b-saes)
- **Trainer**: trainer_2 (BatchTopK, width=65,536, k=80)
- **Hook point**: `resid_post_layer_32`
- **Inference mode**: JumpReLU (fixed scalar threshold, non-batch-dependent)

## Labels

- **`feature_labels.json`** — `{"feature_id": "label", ...}`
- **`feature_labels.csv`** — `feature_id, activation_freq, label`

### Methodology

1. Cached activations over 40M tokens (90% FineWeb + 10% LMSYS-Chat)
2. Applied outlier-token filtering (resid norm > 10× median zeroed)
3. Generated labels using Claude Sonnet 4.6 via `sae_auto_interp.DefaultExplainer`
   (40 top-activating examples per feature, 64-token context windows)
4. Validated via detection scoring: **96.9% accuracy** on a 498-feature sample

### Coverage

- 20,000 features labeled (random subsample of 65,440 non-dead features)
- 30 features failed labeling (< 0.2%)
- Frequency range of labeled features: ~1e-5 to ~1e-1

## Usage

```python
import json

with open("feature_labels.json") as f:
    labels = json.load(f)

print(labels["125"])
# "The selected text describes spending the night or sleeping at a location..."
```

## Citation

SAE trained by [Adam Karvonen](https://huggingface.co/adamkarvonen/qwen3-32b-saes).
Labels generated using [sae_auto_interp](https://github.com/your-repo/sae-auto-interp).
