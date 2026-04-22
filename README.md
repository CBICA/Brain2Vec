#Brain2Vec: Deep Learning Radiomic Features for Neuroimaging Analysis

Code for the [Brain2Vec]() paper, a 3D foundation model pre-trained on 74,425
T1-weighted brain MRI scans. In this repo, you can pre-train your own brain2vec
model, fine-tune a Vision Transformer (ViT) using our pre-trained brain2vec
encoder, train your own downstream ML models using brain2vec output embeddings
or perform downstream classification inference using our pre-trained disease
classification models trained on brain2vec embeddings.

* Brain2Vec implementation follows the code of [Masked Autoencoders](https://github.com/facebookresearch/mae) from Meta.

* Vision Transformers implementation follows the code of [timm](https://github.com/huggingface/pytorch-image-models) from huggingface.

* Before using any of our models, make sure that DLICV() and linear registration to a common atlas space is applied on your input.

## Installation
Brain2Vec exists as a PyPI package and can be installed using pip
```bash
pip install Brain2Vec
```

## Brain2Vec pre-training

We trained with two A100 GPUs for 500 epochs with a patience of 20 epochs. For
more training details please read Brain2Vec paper.
Using our code, you can parallelize your training on SLURM cluster environments.
Other environments are supported but not tested, so bugs may appear.

```bash
srun Brain2Vec brain2vec_pretrain --img_size 128 \
                         --masking_type random \
                         --mask_ratio 0.75 \
                         --model brain2vec_base \
                         --in_dir "input_dir/"
                         --batch_size 32 \
                         --out_dir "output_directory/" \
                         --model_name "model name(.pth)" \
                         --device cuda \
                         --lr 0.001 \
                         --weight_decay 0.05 \
                         --optimizer AdamW \
                         --epochs 500 \
                         --patience 20 \
                         --distributed \
                         --verbose \
```

## Vision Transformer (ViT) fine-tuning

One way of using Brain2Vec is for encoder fine-tuning. The pre-trained encoder
weights can serve as initialization for Vision Transformer (ViT) models applied
to downstream tasks like disease classification or brain age regression.

The only thing you need is a csv with MRID and target columns.

```bash
srun Brain2Vec vit_finetune --img_size 128 \
                         --mode binary \ # or binary, multiclass
                         --model vit_base \
                         --num_classes -1 \ # only for multiclass mode
                         --target disease \ # target to predict on label dictionary
                         --pretrain weights/NiChart_Brain2Vec_base_patch8_dec512d8b.pth \
                         --in_dir "input_dir/"
                         --label_dict "csv with MRID and target columns" \
                         --batch_size 32 \
                         --out_dir "output_directory/" \
                         --model_name "model name (.pth)" \
                         --device cuda \
                         --lr 0.001 \
                         --weight_decay 0.05 \
                         --optimizer AdamW \
                         --epochs 200 \
                         --patience 20 \
                         --distributed \
                         --verbose \
```

## Inference

### Feature extraction
