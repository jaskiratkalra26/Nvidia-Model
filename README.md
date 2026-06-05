# Nvidia-Model

This repository contains a standalone Python script to process a video frame-by-frame and detect surgical instruments using the `nvidia/LocateAnything-3B` Vision-Language Model.

## Requirements

The requirements needed to run this script are provided in `requirements.txt`.
A dedicated NVIDIA GPU is highly recommended due to the 3-Billion parameter size of the model.

## Usage

```bash
python locate_video.py --input path_to_your_video.mp4
```
