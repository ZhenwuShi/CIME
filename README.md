<p align="center">
  <img src="cover.png" width="100%" alt="CIME Cover">
</p>

<h1 align="center">Spatial-Temporal Synergy: Balancing Change and Invariance in Text-Driven 3D Human Motion Editing</h1>

<p align="center">
  <a href='https://github.com/ZhenwuShi/CIME'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white' alt='GitHub'>
  </a>
</p>

---

## Environment & Data Setup

Our data and environment follow [SimMotionEdit](https://github.com/lzhyu/SimMotionEdit.git) and [OmniME](https://github.com/rocket-ycyer/OmniME). Please refer to [motionfix](https://github.com/atnikos/motionfix) to download the dataset and set up the environment, then place the data in the corresponding locations.

The STANCE dataset can be downloaded from [Baidu Pan](https://pan.baidu.com/s/10it5Ma1AqV9s8fFKzWFW5w?pwd=qk8y) (code: `qk8y`).

## Pretrained Checkpoint

You can download our pretrained checkpoint from [Baidu Pan](https://pan.baidu.com/s/1cotZ1snU-oeF7T-fmigyfA?pwd=3a55) (code: `3a55`). The directory structure follows the same layout as [SimMotionEdit](https://github.com/lzhyu/SimMotionEdit.git).

## Training

```bash
python -u train.py --config-name="train_cls_arch" experiment=cls_arch run_id=CIME
```

## Evaluation

#### Step 1: Extract samples

```bash
python motionfix_evaluate.py \
    folder=/path/to/exp \
    guidance_scale_text_n_motion=2.0 \
    guidance_scale_motion=2.0 \
    data=motionfix
```

#### Step 2: Compute metrics

```bash
python compute_metrics.py folder=/path/to/exp/samples/npys
```

## Demo

```bash
python demo.py \
    folder=/path/to/exp \
    guidance_scale_text_n_motion=2.0 \
    guidance_scale_motion=2.0 \
    data=motionfix
```

## Acknowledgements

Our code is built upon [OmniME](https://github.com/rocket-ycyer/OmniME), and further based on [SimMotionEdit](https://github.com/lzhyu/SimMotionEdit.git) and [motionfix](https://github.com/atnikos/motionfix).

## License

This code is distributed under an MIT LICENSE. We also include the LICENSE of motionfix in this repo. Other third-party datasets and software are subject to their respective licenses.
