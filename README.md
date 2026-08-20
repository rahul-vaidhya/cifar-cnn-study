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

| Model | Augmentation | Trainable params | Seeds | Best test acc (%) |
|---|---|---|---|---|
| ResNet-18 (transfer) | yes | 11,181,642 | 2 | **92.10 +/- 0.27** |
| Scratch CNN | no | 814,570 | 2 | 84.84 +/- 0.04 |
| Scratch CNN | yes | 814,570 | 2 | 84.05 +/- 0.26 |

Figures are in `results/figures/` and the per-class breakdown is in
`results/per_class_report.txt`.

### What I found

Pretraining was worth about 7.3 points - 92.10% against 84.84% - but it needed 11.2M
trainable parameters to get there against 814k for the small CNN. That is roughly 14x
the parameters for an 8% relative improvement, which felt like less of a win than I
expected before running it.

The surprise was augmentation. I expected it to help the scratch model by several points
and instead it came out slightly worse: 84.05% against 84.84%. The seed spread was tiny
(0.04 and 0.26), so the difference is real and not noise.

Looking at the training curves explained it. Over the final five epochs the un-augmented
model gained 2.35 points and was clearly flattening out, while the augmented model gained
6.83 points and was still climbing steeply when it hit the epoch limit. The train-test gap
tells the same story: +6.16 points without augmentation, meaning it was memorising the
training set, and -1.02 points with it, meaning test accuracy was actually higher than
training accuracy because the augmented training images are harder than the clean test
images. So the augmented model was nowhere near finished at 15 epochs. What I actually
measured was which model converges faster under a fixed epoch budget, not which one ends
up better. To answer the question I thought I was asking, I would need to train both to
convergence rather than to a fixed epoch count. The ResNet has the same problem, gaining
6.93 points over its last five epochs, so 92.10% is not its ceiling either.

Cat and dog dominate the errors in every configuration. In the scratch model they account
for 251 confusions between them (127 cats called dogs, 124 dogs called cats); the ResNet
cuts that to 164. The ResNet also cleans up mistakes the scratch model makes that seem
more basic, like 56 birds called deer and 52 frogs called cats, and what it has left over
is more understandable - 29 automobiles called trucks, 25 aeroplanes called ships. Cat is
the worst class in the best run at 0.836 F1, and dog is second at 0.863, while automobile
and ship both sit above 0.95.

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
