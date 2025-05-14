# dynamic-object-detection

#### MIT 6.8300 Spring 2025 Final Project

### Website: https://andyli23.github.io/dynamic-object-detection/

#### Dependencies

```
sudo apt-get install ffmpeg x264 libx264-dev
git clone https://github.com/mbpeterson70/robotdatapy && cd robotdatapy && pip install .
pip install -e .
```

#### Requirements

Tested on a system with an i9-14900HX, GeForce RTX 4090 Laptop GPU (16GB), 32GB RAM. May not work on systems
with less memory, even if `batch_size` is decreased.

#### Demo

To run the evaluation data in our blog, download the following rosbags:
[hamilton data](https://drive.google.com/file/d/1kZmhye7E61mLJtyaEFTm_aBValKu3VF5/view?usp=sharing) (ROS1), 
[ground truth](https://drive.google.com/drive/folders/1qGDTkIi9izoh6WXzFa-ODQmevd7g-kpr?usp=drive_link) (ROS2)

```
export BAG_PATH=/path/to/hamilton_data.bag
export RAFT=/path/to/dynamic-object-detection/RAFT/
python3 dynamic_object_detection/main.py -p config/hamilton_final.yaml
```

The code for evaluation metrics is in `eval/eval.ipynb`. Change the following lines in the second cell:
```
os.environ['BAG_PATH'] = os.path.expanduser('/path/to/hamilton_data.bag')
gt_bag = '~/path/to/gt_data/'
```
Then run the entire notebook. Outputs will be printed at the bottom.

*Note*: All operations assume undistorted images. Our data is already undistorted.