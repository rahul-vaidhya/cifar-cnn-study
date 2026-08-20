"""Aggregate results/*.json into a comparison table, curves and confusion matrices."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs("results/figures", exist_ok=True)
runs = [json.load(open(p)) for p in sorted(glob.glob("results/*.json"))]
if not runs:
    raise SystemExit("No results found. Run the experiments first.")

# ---- summary table (mean +/- std across seeds) -------------------------------
groups = {}
for r in runs:
    groups.setdefault((r["model"], r["augment"], r.get("freeze_backbone", False)), []).append(r)

lines = ["| Model | Augmentation | Frozen backbone | Trainable params | Seeds | Best test acc (%) |",
         "|---|---|---|---|---|---|"]
for (model, aug, frz), rs in sorted(groups.items()):
    accs = np.array([r["best_test_acc"] for r in rs])
    lines.append(f"| {model} | {'yes' if aug else 'no'} | {'yes' if frz else 'no'} | "
                 f"{rs[0]['trainable_params']:,} | {len(rs)} | "
                 f"{accs.mean():.2f} +/- {accs.std():.2f} |")
table = "\n".join(lines)
open("results/summary.md", "w").write(table + "\n")
print(table)

# ---- training curves --------------------------------------------------------
plt.figure(figsize=(11, 4))
for i, key in enumerate(["test_acc", "test_loss"]):
    plt.subplot(1, 2, i + 1)
    for r in runs:
        if r["seed"] != runs[0]["seed"]:
            continue
        plt.plot(range(1, len(r["history"][key]) + 1), r["history"][key], label=r["tag"])
    plt.xlabel("epoch"); plt.ylabel(key.replace("_", " ")); plt.grid(alpha=.3)
    if i == 0:
        plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/figures/curves.png", dpi=150)
print("wrote results/figures/curves.png")

# ---- confusion matrix + per-class report for the best run -------------------
best = max(runs, key=lambda r: r["best_test_acc"])
d = np.load(f"results/{best['tag']}_preds.npz")
cm = confusion_matrix(d["labels"], d["preds"])
classes = best["classes"]

plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xticks(range(10), classes, rotation=45, ha="right"); plt.yticks(range(10), classes)
thr = cm.max() / 2
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                 color="white" if cm[i, j] > thr else "black")
plt.title(f"Confusion matrix - {best['tag']} ({best['best_test_acc']:.2f}%)")
plt.ylabel("true"); plt.xlabel("predicted")
plt.tight_layout(); plt.savefig("results/figures/confusion_matrix.png", dpi=150)

rep = classification_report(d["labels"], d["preds"], target_names=classes, digits=3)
open("results/per_class_report.txt", "w").write(f"Best run: {best['tag']}\n\n{rep}\n")
print(rep)

off = cm.copy(); np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"Most confused pair: true '{classes[i]}' predicted as '{classes[j]}' ({off[i, j]} times)")
