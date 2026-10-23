# MusiPainter

## Introduction
MusiPainter is a thesis project for generating images from music tracks. This implementation is based on the existing [Music_to_figurative_art](https://github.com/AlfredoBaione/Music_to_figurative_art) model, adapted and optimized for processing the [Museart](https://www.kaggle.com/datasets/alfredobaione/museart) dataset. The system supports two fusion architectures: Early Fusion and Cross-Attention, which combine audio features with text embeddings to generate artistic images that reflect the input music's characteristics and motifs.

## Execution Environment
All development and execution was performed on Kaggle cloud environment, leveraging GPU acceleration for efficient training and inference.

## Models
- **BEATs Model**: [https://huggingface.co/camenduru/beats/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt](https://huggingface.co/camenduru/beats/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt)
- **Stable Diffusion**: `sd2-community/stable-diffusion-2-1`

## Usage

### Preprocessing

```bash
# Preprocess audio embeddings
python modules/preprocess/preprocess_audio_embeddings_colab.py \
    --audio_dir ./Museart/ \
    --output_dir ./output/audio_embeddings/ \
    --beats_checkpoint models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
    ...

# Preprocess image latents
python modules/preprocess/preprocess_image_latents_colab.py \
    --image_dir ./Museart/images/ \
    --output_dir ./output/image_latents/ \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2
    ...
```

### Training

#### Early Fusion
```bash
python musipainter_train.py \
    --architecture Musipainter-EF \
    --data_dir ./Museart/ \
    --embeddings_dir ./audio_embeddings/ \
    --train_batch_size 2 \
    --max_train_steps 20000
```

#### Cross-Attention
```bash
python musipainter_train.py \
    --architecture Musipainter-CA \
    --data_dir ./Museart/ \
    --embeddings_dir ./audio_embeddings/ \
    --train_batch_size 8 \
    --max_train_steps 20000
```

### Inference

#### Early Fusion
```bash
python musipainter_test.py \
    --architecture Musipainter-EF \
    --data_dir ./Museart/ \
    --embeddings_dir ./audio_embeddings/ \
    --learned_embeds ./output/learned_embeds.safetensors
```

#### Cross-Attention
```bash
python musipainter_test.py \
    --architecture Musipainter-CA \
    --data_dir ./Museart/ \
    --embeddings_dir ./audio_embeddings/ \
    --learned_embeds ./output/learned_embeds.safetensors
```