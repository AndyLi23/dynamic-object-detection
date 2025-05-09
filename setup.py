from setuptools import setup, find_packages

setup(
    name="dynamic-object-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy<2",
        "opencv-python",
        "torch<2.1.0",
        "torchvision",
        "matplotlib",
        "tensorboard",
        "scipy",
        "opencv-python",
        "dataclasses",
        "open3d",
        "tqdm"
    ],
)