# Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI

Code for the [Brain2Vec]() paper, a 3D foundation model pre-trained on 74,425
T1-weighted brain MRI scans. In this repo, you can pre-train your own brain2vec
model, fine-tune a Vision Transformer (ViT) using our pre-trained brain2vec
encoder, train your own downstream ML models using brain2vec output embeddings
or perform downstream classification inference using our pre-trained disease
classification models trained on brain2vec embeddings.

![brain2vec-example](assets/brain2vec_example.png)
*Left: Masked input | Middle: Reconstruction (Z-normalized) | Right: Ground truth*

## Fine-tuning with Brain2Vec

By fine-tuning with our pre-trained model, we achieved both better accuracy and
convergence speed on downstream deep learning tasks

![ba-training-results](assets/ba_training_results.png)

## Using Brain2Vec as a feature extractor

We proved that using Brain2Vec embeddings we get competitive classification
performance across multiple disease groups as well as better accuracy than
DLMUSE ROI's for brain age regression and on-par accuracy with DL fine-tuned
models.

![dx-classification-results](assets/dx_classification_results.png)
![ba-regression-results](assets/ba_regression_results.png)

We also proved that foundation models focus on limited but important set of brain
regions. Manipulating a subset of important latent features produces monotonic,
region-specific effects on the reconstructed anatomy.

![cross-region-correlation](assets/region_profile_lines.png)

* Brain2Vec implementation follows the code of [Masked Autoencoders](https://github.com/facebookresearch/mae) from Meta.

* Vision Transformers implementation follows the code of [timm](https://github.com/huggingface/pytorch-image-models) from huggingface.

* You can use Brain2Vec -- for both feature extraction and disease classification/brain age regression -- on [NiChart](https://cloud.neuroimagingchart.com) by simply dragging and dropping your NIfTI files. **No GPU or installation required**.

* Before using any of our models locally, make sure that the input is
  skull-stripped and linearly registered to a common atlas space. We use
  [NiChart\_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE) for skull-stripping (DLICV folder)
  and [flirt](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/index.html)
  for registration.

  ```bash
  NiChart_DLMUSE -i {dir with nifti files} -o ... ---refaced_data
  flirt -i {DLICV aligned output} -o ...
  ```

If you use our model, make sure to cite our work:
```
@Article{Brain2Vec2026,
  author  = {Spiros Maggioros and Guray Erus and Gareth Harman and George Aidinis},
  journal = {arXiv:},
  title   = {Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI},
  year    = {2026},
}
```

## Installation
Brain2Vec exists as a PyPI package and can be installed using pip
```bash
pip install Brain2Vec
```

## Brain2Vec pre-training

We trained with two A100 GPUs for 500 epochs with a patience of 20 epochs. For
more training details please read the Brain2Vec paper.
Using our code, you can parallelize your training on SLURM cluster environments.
Other environments are supported but not tested, so bugs may appear.

```bash
srun Brain2Vec brain2vec_pretrain --img_size 128 \
                         --masking_type random \
                         --mask_ratio 0.75 \
                         --model brain2vec_base \
                         --in_dir "input_dir/"
                         --batch_size 32 \
                         --out_dir "output_dir/" \
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
                         --mode binary \
                         --model vit_base \
                         --label_dict "csv with MRID and target columns" \
                         --target "target column" \
                         --pretrain "pre-trained model weights (.pth)" \
                         --in_dir "input_dir/"
                         --batch_size 32 \
                         --out_dir "output_dir/" \
                         --model_name "model name (.pth)" \
                         --device cuda \
                         --lr 0.001 \
                         --weight_decay 0.05 \
                         --optimizer AdamW \
                         --epochs 200 \
                         --patience 20 \
                         --distributed \
                         --verbose
```

## Inference

One other way of using Brain2Vec is for feature extraction. We can use the
Brain2Vec frozen encoder to produce 768-dimensional embeddings from T1 scans.
This compact numerical representation serves as input for lightweight downstream
classifiers, such as SVM or MLP models.

```bash
python inference.py brain2vec_inference --model brain2vec_base \
                                    --img_size 128 \
                                    --model_weights "model weights (.pth)" \
                                    --in_dir "path to directory with niftii scans" \
                                    --out_json "name of json file with output embeddings" \
                                    --device cuda

python inference.py vit_inference --model vit_base \
                                --num_classes 1 \
                                --img_size 128 \
                                --mode regression \
                                --model_weights "model weights (.pth)" \
                                --in_dir "input directory with niftii scans" \
                                --out_csv "output csv file with predictions" \
                                --device cuda
```

## Use Brain2Vec embeddings for downstream tasks

We provide a simple pipeline that enable users to train their own ML models
using output embeddings from Brain2Vec.

```bash
python train_svm.py dx_classification --cn_embeddings "path to your CN embeddings" \
                                      --dx_embeddings "path to your DX embeddings"

python train_svm.py ba_regression --embeddings "path to your embeddings" \
                                  --label_dict "path to a CSV that contains MRID and Age columns"
```
