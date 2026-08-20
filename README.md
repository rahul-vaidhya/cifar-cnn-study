# CIFAR-10: CNN from Scratch vs. Transfer Learning

A controlled comparison of a small convolutional network trained from scratch against a
fine-tuned ResNet-18, with an ablation on data augmentation. Every configuration is run
across 3 random seeds and reported as mean +/- standard deviation, because single-run
accuracy numbers on CIFAR-10 vary by more than most of the differences people report.

## Research questions

1. How much accuracy does ImageNet pretraining buy on CIFAR-10, and at what cost in
   trainable parameters?
2. How much of the scratch model's gap is closed by data augmentation alone?
3. Which classes remain hard after both, and are the errors systematic?

## Experimental setup

| | Scratch CNN | ResNet-18 (transfer) |
|---|---|---|
| Architecture | 3 conv blocks (32/64/128), batch norm, dropout 0.5 | torchvision ResNet-18, ImageNet weights, new FC head |
| Input | 32x32 | 64x64 (upsampled) |
| Optimiser | SGD, lr 0.01, momentum 0.9, weight decay 5e-4 | Adam, lr 1e-3 |
| Schedule | Cosine annealing | Cosine annealing |
| Epochs | 20 | 10 |
| Seeds | 0, 1, 2 | 0, 1, 2 |

Augmentation, where enabled: random crop with 4px padding + random horizontal flip.
No test-time augmentation. The test set is used only for evaluation, never for
model selection across configurations.

## Results

<!-- Paste the contents of results/summary.md here after running the experiments. -->

_Run `./run_all.sh`, then paste `results/summary.md` into this section._

**Figures:** `results/figures/curves.png`, `results/figures/confusion_matrix.png`
**Per-class breakdown:** `results/per_class_report.txt`

### Observations

<!-- Fill in after running. Write what you actually observed, including anything that
     surprised you or contradicted your expectation. -->

## Reproducing

```bash
pip install -r requirements.txt
./run_all.sh              # full grid, ~40 min on a Colab T4
```

Single run:

```bash
cd src
python train.py --model scratch --epochs 20 --augment --seed 0
python train.py --model resnet18 --epochs 10 --augment --seed 0
python report.py
```

## Repository layout

```
src/data.py      CIFAR-10 loaders; augmentation behind a flag so it can be ablated
src/models.py    SmallCNN (from scratch) and ResNet-18 transfer head
src/train.py     Training loop, per-epoch eval, JSON metrics + saved predictions
src/report.py    Aggregates runs into a table, curves, confusion matrix, per-class report
run_all.sh       Full experiment grid across 3 seeds
results/         Per-run JSON, predictions, figures (generated)
```

## Limitations

- CIFAR-10 at 32x32 is a small, well-studied benchmark; results here should not be
  read as evidence about performance on higher-resolution or domain-specific data.
- ResNet-18 sees 64x64 upsampled inputs, so the comparison is not parameter-matched;
  it measures "pretrained backbone + upsampling" as a package, not pretraining alone.
- Hyperparameters were set to sensible defaults rather than tuned per configuration.
  A tuned scratch model would close some of the reported gap.
- 3 seeds is enough to show that differences exceed seed noise, not enough for a
  tight confidence interval.

## Next steps

- Iterative magnitude pruning (Lottery Ticket Hypothesis) on the scratch CNN, to test
  how far the network can be sparsified before accuracy degrades.
- Temporal modelling: extend the evaluation methodology to a video or time-series task.
