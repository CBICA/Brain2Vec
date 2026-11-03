# Brain2Vec pre-processing

## Installation
We use [NiChart\_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE) for skull-stripping (s3 masked folder)
and [flirt](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/index.html)
for registration, though you can use any skull-stripping model and
achieve comparable results.

You can install NiChart_DLMUSE from source:
```bash
git clone https://github.com/CBICA/NiChart_DLMUSE
cd NiChart_DLMUSE
pip install -e .
```

To install flirt, please read flirt's [documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html).

> [!TIP]
> We suggest to use a different python virtualenv for preprocessing and Brain2Vec
> inference to avoid dependency issues with DLMUSE.
> We also suggest to use multithreading for flirt alignment (via a thread pool)
> to parallelize the second part of preprocessing. Multithreading is not
> provided in the code because every cluster environment is different, code
> provided works on every environment.

## Run pre-processing pipeline

To run the pre-processing pipeline, just do:
```bash
python brain2vec_preprocess.py --in_dir ... --out_dir ... --device cpu/cuda
```

In out_dir/brain2vec_aligned_scans, you will find the Brain2Vec pre-processed
scans ready for inference.
