from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


setup(
    name="restolo-studio",
    version="0.1.0",
    description="Unified Nanonis control and ReSTOLO analysis desktop application",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "PyQt5",
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "Pillow",
        "PyYAML>=5.3.1",
        "scipy>=1.4.1",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "tqdm>=4.41.0",
        "tensorboard>=2.4.1",
        "seaborn>=0.11.0",
        "pandas",
        "matplotlib>=3.2.2",
        "thop",
        "pycocotools>=2.0",
    ],
    entry_points={
        "console_scripts": [
            "restolo-studio=main:main",
        ]
    },
)
