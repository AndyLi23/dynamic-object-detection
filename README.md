# dynamic-object-detection
For 6.8300 sp25

#### Dependencies

```
pip install 'numpy<2' torch torchvision matplotlib tensorboard scipy opencv-python dataclasses open3d tqdm
sudo apt-get install ffmpeg x264 libx264-dev
git clone https://github.com/mbpeterson70/robotdatapy && cd robotdatapy && pip install .
```

#### Demo

with Kimera-Multi:

```
export KMD=/path/to/KMD/
export RAFT=/path/to/RAFT/
python3 src/main.py --params config/kmd_demo.yaml
```

*Note*: All operations assume undistorted images. KMD data is already undistorted.