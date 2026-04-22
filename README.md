# Brain2Vec: Deep Learning Radiomic Features for Neuroimaging Analysis

Code for the [Brain2Vec]() paper, a 3D foundation model pre-trained on 74,425
T1-weighted brain MRI scans. In this repo, you can pre-train your own brain2vec
model, fine-tune a Vision Transformer (ViT) using our pre-trained brain2vec
encoder, train your own downstream ML models using brain2vec output embeddings
or perform downstream classification inference using our pre-trained disease
classification models trained on brain2vec embeddings.

* Brain2Vec implementation follows the code of [Masked Autoencoders](https://github.com/facebookresearch/mae) from Meta.

* Vision Transformers implementation follows the code of [timm](https://github.com/huggingface/pytorch-image-models) from huggingface.

## Brain2Vec pre-training

To perform pre-training, you can just do:

```bash

```

