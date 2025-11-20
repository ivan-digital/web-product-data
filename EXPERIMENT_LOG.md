# Embedding Classifier Experiments

## 2025-11-17 - Baseline + Scheduler Trial (completed)
- **Command**: `poetry run python category_classification/train_embedding_classifier.py --data ./lspc_dataset_full --out ./embedding_classifier_prodcat --resume ./embedding_classifier_prodcat/checkpoint-epoch3.pt --epochs 5 --batch_size 254 --grad_accum 2 --lr_schedule cosine --lr_warmup_steps 3000 --lr_min_scale 0.05 --save_every 1 --amp`
- **Result**: Validation macro-F1 ~0.932 / micro-F1 ~0.948 at epoch 3. Test language metrics highlight the English skew (~844k/952k rows): `en` macro-F1 0.948 / micro-F1 0.963, while `de` 0.809, `fr` 0.827, `nl` 0.768, `it` 0.772.
- **Issue**: Strong language imbalance - non-English classes lag despite solid overall accuracy.

## 2025-11-18 - V2 Residual Head + Label Smoothing (completed)
- **Command**: `poetry run python category_classification/train_embedding_classifier_v2.py --data ./lspc_dataset_full --out ./embedding_classifier_prodcat_v2 --resume ./embedding_classifier_prodcat_v2/checkpoint-epoch2.pt --epochs 5 --batch_size 128 --grad_accum 4 --lr_schedule cosine --lr_warmup_steps 2000 --lr_min_scale 0.05 --label_smoothing 0.05 --focal_gamma 1.5 --amp`
- **Result**: Validation macro-F1 plateaued near 0.9316; test macro-F1 0.9300 / micro-F1 0.9477.
- **Language metrics (test)**: `en` 0.944 / `de` 0.748 / `fr` 0.800 / `es` 0.781 / `ja` 0.827.
- **Next steps**: Try language-aware sampling or per-language loss weights; consider a short non-English-only fine-tune; experiment with lower focal gamma.

## 2025-11-19 - V2 Wide Head + No Focal (partial)
- **Command**: `poetry run python category_classification/train_embedding_classifier_v2.py --data ./lspc_dataset_full --out ./embedding_classifier_prodcat_v2_gamma0 --epochs 8 --batch_size 256 --grad_accum 4 --lr_schedule cosine --lr_warmup_steps 2000 --lr_min_scale 0.01 --weight_decay 0.02 --classifier_hidden 1024 --classifier_dropout 0.2 --label_smoothing 0.05 --focal_gamma 0 --amp`
- **Progress**: Training reached epoch 6 (best validation macro-F1 ~0.9326 at epoch 4). 
- **Observation**: Even with wider head, stronger dropout, and no focal loss, validation macro-F1 stayed roughly in the 0.932-0.933 band, suggesting we need more drastic measures (sampling/contrastive) to lift minority languages.

## 2025-11-19 - V2 Baseline Head (English-focused, completed)
- **Command**: `poetry run python category_classification/train_embedding_classifier_v2.py --data ./lspc_dataset_full --out ./embedding_classifier_prodcat_v2_en --epochs 5 --batch_size 256 --grad_accum 2 --lr_schedule cosine --lr_warmup_steps 2000 --lr_min_scale 0.05 --weight_decay 0.01 --classifier_hidden 384 --classifier_dropout 0.1 --label_smoothing 0.05 --focal_gamma 1.5 --amp`
- **Result**: Validation macro-F1 peaked at 0.9327; test macro-F1 0.9315 / micro-F1 0.9487.
- **Language metrics (test)**: `en` 0.946 / `de` 0.753 / `fr` 0.809 / `es` 0.783 / `ja` 0.830.
- **Notes**: Confirms that focusing on English gives a small boost for EN while other languages stay roughly the same.

## 2025-11-20 - V3 Contrastive Head (in progress)
- **Command**:
  ```
  poetry run python category_classification/train_embedding_classifier_v3.py \
    --data ./lspc_dataset_full \
    --out ./embedding_classifier_prodcat_v3 \
    --epochs 10 --batch_size 256 --grad_accum 4 \
    --lr_schedule cosine --lr_warmup_steps 3000 --lr_min_scale 0.01 \
    --classifier_hidden 512 --classifier_dropout 0.2 \
    --contrastive_dim 256 --contrastive_weight 0.1 --contrastive_temp 0.05 \
    --label_smoothing 0.05 --focal_gamma 0 --amp
  ```
- **Goal**: Add a SimCSE-style projection head and optimise `L = L_cls + lambda * L_contrastive` so samples sharing the same label/language cluster, with the hope of lifting minority-language macro-F1.
- **Notes**:
  - New files: `embedding_classifier_v3.py`, `train_embedding_classifier_v3.py`.
  - Current implementation uses label-based positives inside each batch; adjust `--contrastive_weight`/`--contrastive_temp` as needed.
  - Optional future work: add KL regularisation to keep English performance aligned with the best v2 model.
