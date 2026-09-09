# MusiPainter
Thesis Project that follow the model Musipainter to create image based on audio track and artistic motif

Executed by actual modules:
- BEATs Model: https://huggingface.co/camenduru/beats/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
- Stable Diffusion: sd2-community/stable-diffusion-2-1
- Various Library: [torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1, xformers, huggingface-hub==0.22.2, transformers==4.38.2, diffusers==0.25.1, datasets==2.18.0, peft==0.8.2, accelerate, ftfy, tensorboard, opencv-python, Pillow, pandas, soundfile, safetensors, timm, scikit-learn]

# Uso

## Train
```bash
# Early Fusion
python musipainter_train.py --architecture Musipainter-EF \
    --data_dir ./Museart/ \
    --embeddings_dir ./audio_embeddings/ \
    --train_batch_size 2 \
    --max_train_steps 20000
    ...

# Cross-Attention
python musipainter_train.py --architecture Musipainter-CA \
    --data_dir ./Museart/ \
    --embeddings_dir ./audio_embeddings/ \
    --train_batch_size 8 \
    --max_train_steps 20000
    ...
```
## Test
```bash
# Early Fusion
python musipainter_test.py --architecture Musipainter-EF \
    --data_dir ./Museart/ \
    --embeddings_dir ./audio_embeddings/ \
    --learned_embeds ./output/learned_embeds.safetensors
    ...

# Cross-Attention
python musipainter_test.py --architecture Musipainter-CA \
    --data_dir ./Museart/ \
    --embeddings_dir ./audio_embeddings/ \
    --learned_embeds ./output/learned_embeds.safetensors
    ...
```