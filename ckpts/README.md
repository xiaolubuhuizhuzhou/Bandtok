# Checkpoints

The LM configs expect a trained tokenizer here:

```
ckpts/bandtok.pt        # tokenizer used by configs/lm/*.json
ckpts/bandtok_1d.pt     # residual variant used by the BandTok-1D ablation
```

Either train the tokenizer yourself (see the README) and copy or symlink the
checkpoint here:

```bash
ln -s "$PWD/runs/bandtok/checkpoints/last.pt" ckpts/bandtok.pt
```

or point `model.pretransform.config.ckpt_path` in the LM config at wherever your
checkpoint already lives — absolute paths, paths relative to the config file, and
paths relative to the repository root all work.

A checkpoint written by `train.py` contains the model weights, the optimizer and
scheduler state, the step counter, and the config the run used, so it can be used
both for resuming (`--resume`) and for inference.
