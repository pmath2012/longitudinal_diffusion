This codebase has been built upon the codebase for improved-diffusion [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

# Usage

Please see the original repository for installation details

You would require a pre-trained model for bitemporal segmentation of generated images. The repo uses a Bi-temporal UNeT from the glasses repository.

please run the following to train the diffusion model:
```
export CUDA_VISIBLE_DEVICES=1
python train_ms.py --root_dir /path/to/data/ \
                    --csv_file training.csv \
                    --batch_size 4 \
                    --lr 1e-4 \
                    --save_interval 500 \
                    --log_interval 10 \
                    --ema_rate 0.9999 \
                    --use_fp16 False \
                    --world_size 1 \
                    --class_cond True \
                    --rescale_learned_sigmas False \
                    --learn_sigma False \
                    --segmentation_model /path/to/trained/model/ckpoint.pt
```
