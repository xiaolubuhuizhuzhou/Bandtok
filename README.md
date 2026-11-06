# BandTok

Official implementation of **"Modeling Music as a Time-Frequency Image: A 2D Tokenizer for Music Generation."**

BandTok is a generation-oriented 2D Mel-spectrogram tokenizer that represents each frame with Mel-frequency band tokens from a single shared codebook. A companion
autoregressive LM models the flattened grid with **2D RoPE**, so temporal and
spectral locality survive flattening.

[![arXiv](https://img.shields.io/badge/arXiv-2503.10522-brightgreen)](https://arxiv.org/abs/2605.15831)
[![Hugging Face Model](<https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue>)](https://huggingface.co/xlbhzz/bandtok-model)
[![Demo Page](https://img.shields.io/badge/Demo-Page-brightgreen)](https://xiaolubuhuizhuzhou.github.io/bandtok_demo/)

## Install

```bash
git clone https://github.com/xiaolubuhuizhuzhou/Bandtok.git
cd Bandtok
pip install -e .                 # or: pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional, for LM training throughput
```

Two pretrained models are fetched at runtime: the frozen **BigVGAN-v2** vocoder
(`nvidia/bigvgan_v2_44khz_128band_512x`) and the **T5-base** text encoder. Set
`BANDTOK_VOCODER_PATH` and `HF_HOME` to use local copies on an offline cluster.

## Data

Every dataset is described by a JSONL manifest, one object per line:

```json
{"path": "relative/or/absolute.wav", "caption": "a mellow jazz trio", "duration": 212.4}
```

Build one from a directory of audio:

```bash
python scripts/build_manifest.py --audio-dir /data/mtg-jamendo/audio \
    --relative-to /data/mtg-jamendo/audio --output data/manifests/mtg_jamendo.jsonl
```

Then edit the `root` / `manifest` fields in `configs/data/*.json`.

## Train the tokenizer

```bash
torchrun --nproc_per_node 8 train.py \
    --model-config configs/tokenizer/bandtok.json \
    --data-config configs/data/tokenizer_train.json \
    --val-data-config configs/data/valid_songdescriber.json \
    --run-dir runs/bandtok --batch-size 128 --num-workers 8
```

(or `bash train.sh tokenizer`, which wraps the same command, picks the machine-local
data config when one exists, and takes `MODEL_CONFIG`, `BATCH_SIZE`, `RUN_DIR`,
`MAX_STEPS`, `NNODES`, `MASTER_ADDR` … from the environment. Pointing `MODEL_CONFIG`
at a different config gets its own run directory automatically.)

## Train the language model

```bash
torchrun --nproc_per_node 8 train.py \
    --model-config configs/lm/bandtok.json \
    --data-config configs/data/lm_train.json \
    --run-dir runs/bandtok_lm --batch-size 16 --num-workers 12
```

(or `bash train.sh lm`.)

Point `model.pretransform.config.ckpt_path` in the LM config at your trained tokenizer
checkpoint first — the tokenizer is frozen and only used to produce tokens.

## Generate music

```bash
python scripts/generate.py \
    --model-config configs/lm/bandtok.json --ckpt runs/bandtok_lm/checkpoints/last.pt \
    --prompt "a driving synthwave instrumental with analog bass" \
    --duration 10 --cfg-scale 2.0 --output-dir samples/
```

## Evaluate

```bash
# Reconstruction:
python scripts/evaluate_reconstruction.py \
    --model-config configs/tokenizer/bandtok.json --ckpt runs/bandtok/checkpoints/last.pt \
    --data-config configs/data/valid_musdb_test.json --num-segments 1000

# Round-trip a file through the tokenizer
python scripts/reconstruct.py \
    --model-config configs/tokenizer/bandtok.json --ckpt runs/bandtok/checkpoints/last.pt \
    --input song.wav --output-dir recons/
```


## Citation

```bibtex
@inproceedings{cheng2026bandtok,
  title     = {Modeling Music as a Time-Frequency Image: A 2D Tokenizer for Music Generation},
  author    = {Cheng, Yuqing and Ma, Xingyu and Yu, Guochen and Gu, Xiaotao},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME), Grand Challenge},
  year      = {2026}
}
```

Please also cite MelTok, whose tokenizer architecture BandTok borrows from:

```bibtex
@article{li2025meltok,
  title   = {MelTok: 2D Tokenization for Single-Codebook Audio Compression},
  author  = {Li, Jingyi and Zhao, Zhiyuan and Zhang, Zhisheng and Liu, Yunfei and
             Lin, Lijian and Zhu, Ye and Wu, Jiahao and Kong, Qiuqiang and Li, Yu},
  journal = {arXiv preprint arXiv:2510.01903},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.01903}
}
```

## Acknowledgements

BandTok's tokenizer architecture borrows from
[MelTok](https://arxiv.org/abs/2510.01903) (formerly MelCap), and the implementation
builds on [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools),
the [NVIDIA Cosmos tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer),
[BigVGAN](https://github.com/NVIDIA/BigVGAN), [auraloss](https://github.com/csteinmetz1/auraloss),
[vgg_loss](https://github.com/crowsonkb/vgg_loss), [audiocraft](https://github.com/facebookresearch/audiocraft)
and [Transformers](https://github.com/huggingface/transformers). Per-component licenses
and the files they apply to are listed in [`NOTICE`](NOTICE); license texts are in
[`LICENSES/`](LICENSES). This repository is released under the MIT License.
