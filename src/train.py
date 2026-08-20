"""Train one configuration and write metrics + checkpoint to results/."""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from data import get_loaders, CLASSES
from models import build


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = correct = total = 0
    preds_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
        preds_all.append(pred.cpu()); labels_all.append(y.cpu())
    return (loss_sum / total, 100.0 * correct / total,
            torch.cat(preds_all).numpy(), torch.cat(labels_all).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scratch", "resnet18"], required=True)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 64 if a.model == "resnet18" else 32
    lr = a.lr if a.lr is not None else (0.001 if a.model == "resnet18" else 0.01)
    tag = a.tag or f"{a.model}_aug{int(a.augment)}_seed{a.seed}"

    train_loader, test_loader = get_loaders(a.batch_size, a.augment, img_size)
    kw = {"num_classes": 10}
    if a.model == "resnet18":
        kw["freeze_backbone"] = a.freeze_backbone
    model = build(a.model, **kw).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()
    opt = (torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
           if a.model == "resnet18" else
           torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    os.makedirs("results", exist_ok=True)
    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best, t0 = 0.0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); ls = c = n = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x); loss = criterion(out, y)
            loss.backward(); opt.step()
            ls += loss.item() * y.size(0)
            c += (out.argmax(1) == y).sum().item(); n += y.size(0)
        sched.step()
        te_loss, te_acc, preds, labels = evaluate(model, test_loader, device, criterion)
        hist["train_loss"].append(ls / n); hist["train_acc"].append(100.0 * c / n)
        hist["test_loss"].append(te_loss); hist["test_acc"].append(te_acc)
        if te_acc > best:
            best = te_acc
            np.savez(f"results/{tag}_preds.npz", preds=preds, labels=labels)
        print(f"[{tag}] ep {ep:02d}/{a.epochs} "
              f"train {hist['train_acc'][-1]:.2f}% | test {te_acc:.2f}% | best {best:.2f}%")

    json.dump({"tag": tag, "model": a.model, "augment": a.augment,
               "freeze_backbone": a.freeze_backbone, "seed": a.seed,
               "epochs": a.epochs, "lr": lr, "trainable_params": n_params,
               "best_test_acc": best, "final_test_acc": hist["test_acc"][-1],
               "minutes": round((time.time() - t0) / 60, 2), "history": hist,
               "classes": list(CLASSES)},
              open(f"results/{tag}.json", "w"), indent=2)
    print(f"[{tag}] done. best {best:.2f}% in {(time.time()-t0)/60:.1f} min "
          f"({n_params:,} trainable params)")


if __name__ == "__main__":
    main()
