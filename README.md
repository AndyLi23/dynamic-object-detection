# dynamic-object-detection
For 6.8300 sp25

#### Dependencies

```
pip install 'numpy<2' torch torchvision matplotlib tensorboard scipy opencv dataclasses

git clone https://github.com/mbpeterson70/robotdatapy && cd robotdatapy && pip install .
```

#### Demo

with Kimera-Multi outdoor thoth:

```
export $KMD=/path/to/kimera/outdoor/data
export $ROBOT=thoth
python3 src/main.py --params config/kmd_demo.yaml
```