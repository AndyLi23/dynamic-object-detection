# dynamic-object-detection
For 6.8300 sp25

#### Dependencies

```
sudo apt-get install ffmpeg x264 libx264-dev
git clone https://github.com/mbpeterson70/robotdatapy && cd robotdatapy && pip install .
pip install -e .
```

#### Demo

with Kimera-Multi:

```
export KMD=/path/to/KMD/
export RAFT=/path/to/RAFT/
python3 dynamic_object_detection/main.py -p config/kmd_demo.yaml
```

*Note*: All operations assume undistorted images. KMD data is already undistorted.