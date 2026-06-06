# Third-party notices

This software bundles, invokes, or subprocess-launches the following
third-party components. Each retains its original license. Cite the
relevant paper if you publish results that depend on these models.

## Lip-sync — LatentSync 1.6

- **Repo**: https://github.com/bytedance/LatentSync
- **License**: Apache License 2.0
- **Authors**: ByteDance Inc.
- **Paper**: Li et al., *LatentSync: Taming Audio-Conditioned Latent
  Diffusion Models for Lip Sync with SyncNet Supervision*,
  arXiv:2412.09262 (2024).

```bibtex
@article{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models
         for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and Lin, Jingyu
          and Xie, Jinghui and Feng, Jihong and Peng, Bhuwan and
          Zhang, Junhui and Zheng, Wenjia and Pun, Adrian},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}
```

Our `lipsync_test/LatentSync/scripts/train_oneshot.py` is derived
from LatentSync's `scripts/train_unet.py` (Apache 2.0). Per the
license, we have:
- Retained the upstream copyright notice
- Added prominent notices stating the modifications (stripped to
  single-GPU, recon-loss-only, no SyncNet/LPIPS/TREPA/validation)
- Released the modified file under Apache 2.0

## Face-swap — inswapper_128

- **Repo**: https://github.com/deepinsight/insightface
- **License**: MIT (codebase). The inswapper_128.onnx model itself
  has been withdrawn from official InsightFace distribution since
  2023; we do NOT bundle it (see [INSTALL.md](INSTALL.md)).
- **Authors**: InsightFace Team / DeepInsight
- **Paper**: Deng et al., *ArcFace: Additive Angular Margin Loss for
  Deep Face Recognition*, CVPR 2019.

```bibtex
@inproceedings{deng2019arcface,
  title={ArcFace: Additive angular margin loss for deep face
         recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou,
          Stefanos},
  booktitle={CVPR},
  year={2019}
}
```

The InsightFace SDK (`insightface` pip package) and the ONNX
inference runtime are used as imports.

## Face restoration — GFPGAN v1.4

- **Repo**: https://github.com/TencentARC/GFPGAN
- **License**: Apache License 2.0
- **Authors**: Tencent ARC Lab
- **Paper**: Wang et al., *Towards Real-World Blind Face Restoration
  with Generative Facial Prior*, CVPR 2021.

```bibtex
@inproceedings{wang2021gfpgan,
  title={Towards Real-World Blind Face Restoration with Generative
         Facial Prior},
  author={Wang, Xintao and Li, Yu and Zhang, Honglun and Shan, Ying},
  booktitle={CVPR},
  year={2021}
}
```

## Vocal isolation — Demucs

- **Repo**: https://github.com/facebookresearch/demucs
- **License**: MIT
- **Authors**: Meta AI Research (Défossez et al.)
- **Paper**: Défossez, *Hybrid Spectrogram and Waveform Source
  Separation*, MDX Workshop at ISMIR 2021.

```bibtex
@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source
             Separation},
  year={2021}
}
```

## Mask propagation — SAM 2.1

- **Repo**: https://github.com/facebookresearch/sam2
- **License**: Apache License 2.0
- **Authors**: Meta AI Research
- **Paper**: Ravi et al., *SAM 2: Segment Anything in Images and
  Videos*, arXiv:2408.00714 (2024).

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and
          Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and
          Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe
          and Gustafson, Laura and Mintun, Eric and Pan, Junting
          and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu,
          Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and
          Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## Audio features — Whisper

- **Repo**: https://github.com/openai/whisper
- **License**: MIT
- **Authors**: OpenAI
- **Paper**: Radford et al., *Robust Speech Recognition via Large-
  Scale Weak Supervision*, 2022.

```bibtex
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman,
          Greg and McLeavey, Christine and Sutskever, Ilya},
  year={2022},
  publisher={OpenAI}
}
```

## Voice swap — RVC (optional, subprocess only)

- **Repo**: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- **License**: MIT
- **Note**: Used only when the user installs RVC separately. We do
  not ship RVC or its models.

## VAE — stabilityai/sd-vae-ft-mse

- **Repo**: https://huggingface.co/stabilityai/sd-vae-ft-mse
- **License**: CreativeML OpenRAIL-M (derivative use permitted)
- **Note**: Used by LatentSync as the latent space VAE.

---

If we have missed an attribution, please open an issue and we will
fix it.
