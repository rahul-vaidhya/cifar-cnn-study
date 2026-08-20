# CIFAR-10: CNN from scratch vs. transfer learning

I wanted to find out how much ImageNet pretraining actually helps on CIFAR-10, and
how much of that gap I could close just by adding data augmentation to a network I
trained myself. So I set both up properly and compared them.

Every configuration runs on two random seeds and I report the mean and standard
deviation. This turned out to matter: one of the differences I found is smaller than a
point, and without knowing the seed-to-seed spread I would not have been able to say
whether it was real or noise.

## What I compared

|  | Scratch CNN | ResNet-18 (transfer) |
|---|---|---|
| Architecture | 3 conv blocks (32/64/128), batch norm, dropout 0.5 | torchvision ResNet-18, ImageNet weights, new FC head |
| Input size | 32x32 | 64x64 (upsampled) |
| Optimiser | SGD, lr 0.01, momentum 0.9, weight decay 5e-4 | Adam, lr 1e-3 |
| Schedule | Cosine annealing | Cosine annealing |
| Epochs | 15 | 8 |

The ResNet gets fewer epochs because it converges much faster - the backbone already
knows how to extract features, so only the head really needs training.

Where augmentation is enabled it is random crop with 4px padding plus random horizontal
flip. No test-time augmentation. I only look at the test set to evaluate, never to pick
between configurations.

## Results

| Model | Augmentation | Epochs | Trainable params | Seeds | Best test acc (%) |
|---|---|---|---|---|---|
| ResNet-18 (transfer) | yes | 8 | 11,181,642 | 2 | 92.10 +/- 0.27 |
| Scratch CNN | no | 15 | 814,570 | 2 | 84.84 +/- 0.04 |
| Scratch CNN | yes | 15 | 814,570 | 2 | 84.05 +/- 0.26 |
| Scratch CNN | no | 50 | 814,570 | 1 | 85.87 |
| Scratch CNN | yes | 50 | 814,570 | 1 | **89.38** |

Figures are in `results/figures/` and the per-class breakdown is in
`results/per_class_report.txt`.

### What I found

My first run used 15 epochs and augmentation came out slightly *worse* - 84.05% against
84.84%. The seed spread was 0.04 and 0.26, so the difference was real and not noise, but
it contradicted everything I expected.

Looking at the curves suggested why. The un-augmented model gained 2.35 points over its
final five epochs and was flattening out, while the augmented one gained 6.83 and was
still climbing steeply when it hit the epoch limit. So I reran both at 50 epochs to find
out whether augmentation was actually worse or just slower.

It was just slower. At 50 epochs augmentation wins by 3.56 points, 89.38% against 85.87%.
The two curves cross at **epoch 10**, and the augmented model holds the lead from epoch 20
onwards. Both runs are genuinely converged this time - over the last five epochs the
un-augmented model changed by -0.06 points and the augmented one by +0.03.

The overfitting numbers make the mechanism obvious. Without augmentation the model reaches
**99.64% training accuracy against 85.81% test** - a 13.8 point gap, so it has essentially
memorised the training set. With augmentation it reaches 92.54% train against 89.37% test,
a gap of 3.2 points. It never gets to memorise, because it never sees the same image twice.

The part I did not expect is what this does to the transfer learning result. Against the
15-epoch baseline, pretraining looked worth 7.3 points. Against a properly trained baseline
it is worth **2.7 points** - 92.10% against 89.38%. Most of what looked like the benefit of
pretraining was really just my baseline being undertrained. If I had stopped after the
first run I would have drawn a conclusion that was wrong by more than a factor of two,
and I would have had no way of knowing.

Cat and dog dominate the errors in every configuration. In the 15-epoch scratch model they
account for 251 confusions between them (127 cats called dogs, 124 dogs called cats); the
ResNet cuts that to 164. The ResNet also fixes more basic mistakes the scratch model makes,
like 56 birds called deer and 52 frogs called cats, and what it has left is more
understandable - 29 automobiles called trucks, 25 aeroplanes called ships. Cat is the worst
class in the best run at 0.836 F1 and dog is second at 0.863, while automobile and ship
both sit above 0.95.

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

The 50-epoch runs are single-seed, because I had already established from the 15-epoch
runs that seed variation here is small (0.04 to 0.26 points) relative to the 3.56 point
effect I was measuring. That is a reasonable trade but it is still one seed.

The ResNet is the weak point of the comparison now. It ran for 8 epochs and was still
gaining 6.93 points over its last five, so 92.10% is not its ceiling either - which means
the 2.7 point transfer learning advantage I report is itself measured against an
undertrained model. Fixing that properly would mean training everything to convergence,
which is what I should have done from the start.

## Next

I want to try iterative magnitude pruning on the scratch CNN, following the Lottery
Ticket Hypothesis paper, to see how far it can be sparsified before accuracy drops off.
