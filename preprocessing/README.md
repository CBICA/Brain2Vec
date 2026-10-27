# Brain2Vec pre-processing

## Installation
We use [NiChart\_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE) for skull-stripping (s3 masked folder)
and [flirt](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/index.html)
for registration, though you can use any skull-stripping model and
achieve comparable results.

To install NiChart_DLMUSE, you can either install it from source **(Suggested)**:
```bash
git clone https://github.com/CBICA/NiChart_DLMUSE
cd NiChart_DLMUSE
pip install -e .
```

Or install the latest stable version from PyPI:
```bash
pip install NiChart_DLMUSE
```

To install flirt, please read flirt's [documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html).

## Run pre-processing pipeline

To run the pre-processing pipeline, just do:
```bash
python brain2vec_preprocess.py --in_dir ... --out_dir ... --device cpu/cuda
```

In out_dir/brain2vec_aligned_scans, you will find the Brain2Vec pre-processed
scans ready for inference.
