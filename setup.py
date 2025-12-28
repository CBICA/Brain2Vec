from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="NiChart_ViT",
    version="0.0.1",
    description="ViT Training for NiChart",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Spiros Maggioros",
    author_email="software@cbica.upenn.edu",
    maintainer="Spiros Maggioros",
    maintainer_email="Spiros.Maggioros@pennmedicine.upenn.edu",
    download_url="https://github.com/CBICA/NiChart_ViT/",
    url="https://github.com/CBICA/NiChart_ViT/",
    packages=find_packages(exclude=[".github"]),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torchvision",
        "torchio",
        "torchmetrics",
        "tqdm",
        "numpy",
        "pandas",
    ],
    entry_points={"console_scripts": ["NiChart_ViT = NiChart_ViT.__main__:main"]},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    license="By installing/using NiChart_ViT, the user agrees to the following license: See https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html",
    keywords=[
        "deep learning",
        "semantic segmentation",
        "medical image analysis",
    ],
    package_data={"NiChart_ViT": ["VERSION"]},
)
