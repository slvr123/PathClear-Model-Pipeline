# My frozen baseline_v1

I identify the existing saved 20 x 20 model as `baseline_v1`.
I preserve its original files in place and record their SHA-256 fingerprints in `baseline_v1.json`.
I do not retrain or overwrite it.

- Checkpoint: `C:\Users\Asus\OneDrive\Desktop\Code\PathClear\tinyml\fomo\artifacts\diagnosis-center-training-20261004\best.keras`.
- INT8: `C:\Users\Asus\OneDrive\Desktop\Code\PathClear\tinyml\fomo\artifacts\diagnosis-center-export-20261004\model.int8.tflite`.
- Dataset: Cityscapes, 2,975 train and 500 validation images.
- Classes: Person and Vehicle; 160 x 160 RGB input, 20 x 20 grid.
- Precision / recall / F1: 70.65% / 25.43% / 37.40% at confidence 0.60.
- Person: 63.56% / 22.85% / 33.62%.
- Vehicle: 75.16% / 27.07% / 39.80%.
- Recall below 2 px / 2 to below 8 px / at least 8 px: 12.37% / 21.53% / 50.64%.
- INT8 size: 70,736 bytes.
- Training: ImageNet initialization, seed 42, batch 32, object weight 20, horizontal flips, initial learning rate 0.001, maximum 20 epochs, best epoch 19.
- Full training configuration and original history: `baseline_v1.json`.
