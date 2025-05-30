# Pose conditioned generative steganography(PCGS)

<img src="pipeline.jpg" style="zoom:20%;" />

## Requirements

- We only test the code on **Linux**, but **Windows** may be supported as well.
- Python == 3.10,  pytorch == 2.6.0.
- cuda == 11.8  We have done all development and testing using a NVIDIA RTX 4090.

```
conda create -n pcgs python==3.10
conda activate pcgs
pip install -r requirements.txt
```

## Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints usage.

**For COCO pose label**, please download from [Download Link](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip).

**For Viton-Hd data**, please download from [Viton-hd download](https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view?usp=sharing).

To generate images, we use the Automatic111 stable diffusion webui and the Dream Shaper pretrained model based on 	   Stable diffusion 1.5.

## Experiments

