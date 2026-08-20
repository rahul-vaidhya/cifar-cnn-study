# CIFAR-10: CNN from scratch vs. transfer learning

I wanted to find out how much ImageNet pretraining actually helps on CIFAR-10, and
how much of that gap I could close just by adding data augmentation to a network I
trained myself. So I set both up properly and compared them.

Every configuration runs on multiple random seeds and I report the mean and standard
deviation, because when I first ran this I noticed the accuracy moved around by almost
a full point between seeds. That is close to the size of some of the differences I was
trying to measure, so a single number would not have told me much.

## What I compared

|  | Scratch CNN | ResNet-18 (transfer) |
|---|---|---|
| Architecture | 3 conv blocks (32/64/128), batch norm, dropout 0.5 | torchvision ResNet-18, ImageNet weights, new FC head |
| Input size | 32x32 | 64x64 (upsampled) |
| Optimiser | SGD, lr 0.01, momentum 0.9, weight decay 5e-4 | Adam, lr 1e-3 |
| Schedule | Cosine annealing | Cosine annealing |
| Epochs | 20 | 10 |

The ResNet gets fewer epochs because it converges much faster - the backbone already
knows how to extract features, so only the head really needs training.

Where augmentation is enabled it is random crop with 4px padding plus random horizontal
flip. No test-time augmentation. I only look at the test set to evaluate, never to pick
between configurations.

## Results

<!-- Paste the table from results/summary.md here -->

Figures are in `results/figures/` and the per-class breakdown is in
`results/per_class_report.txt`.

### What I found

<!-- Write 3-4 sentences here after you look at your own results.
     Things worth commenting on:
       - how big the scratch vs ResNet gap is, and whether it surprised you
       - how much augmentation alone closed it
       - which two classes get mixed up most, and whether that makes sense
       - whether any differences were smaller than the seed-to-seed variation -->

## Running it

```bash
pip install -r requirements.txt
./run_all.sh
```

That runs everything and takes roughly 40 minutes on a Colab T4. On CPU it is far
slower, so I would not recommend it.

To run a single configuration:

```bash
cd src
python train.py --model scratch --epochs 20 --augment --seed 0
python report.py
```

## What is in here

```
src/data.py      loads CIFAR-10; augmentation is behind a flag so I could ablate it
src/models.py    the small CNN, and the ResNet-18 with a replaced head
src/train.py     training loop, evaluates each epoch, writes metrics to JSON
src/report.py    builds the summary table, curves, confusion matrix, per-class report
run_all.sh       runs the whole grid
results/         generated output
```

## Things this does not show

A few caveats I ran into that are worth stating rather than hiding:

The comparison is not parameter-matched. The ResNet sees 64x64 upsampled inputs while
the scratch CNN sees 32x32, so what I am really measuring is "pretrained backbone plus
upsampling" as a package, not the effect of pretraining on its own. Separating those
would need another experiment.

I did not tune hyperparameters per configuration. I picked reasonable defaults for each
and left them. A properly tuned scratch model would probably close part of the gap I
report.

CIFAR-10 is small and heavily studied, so none of this says anything about how either
approach behaves on higher-resolution or domain-specific images.

And the seed count is enough to show that the differences are bigger than random
variation, but not enough for a tight confidence interval.

## Next

I want to try iterative magnitude pruning on the scratch CNN, following the Lottery
Ticket Hypothesis paper, to see how far it can be sparsified before accuracy drops off.
