import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from synthetic_rescue_dataset import SyntheticRescueBenchmark
from model_rescuenet_x import RescueNetX


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def compute_metrics(y_true, y_prob):
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"] = 0.0
    return out


def topk_zone_accuracy(logits, labels, k=3):
    topk = torch.topk(logits, k=k, dim=1).indices
    return topk.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()


def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    zone_acc, zone_top3 = [], []
    weights = []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)

            probs = torch.sigmoid(out["match_logit"]).cpu().numpy().tolist()
            labels = batch["match_label"].cpu().numpy().tolist()

            y_prob.extend(probs)
            y_true.extend(labels)

            pred_zone = torch.argmax(out["zone_logits"], dim=1)
            zone_acc.append((pred_zone == batch["zone_label"]).float().mean().item())
            zone_top3.append(topk_zone_accuracy(out["zone_logits"], batch["zone_label"], k=3))

            weights.append(out["modality_weights"].mean(dim=0).cpu())

    metrics = compute_metrics(y_true, y_prob)
    mean_weights = torch.stack(weights).mean(dim=0)

    print(
        f"Eval | Acc={metrics['acc']:.3f} "
        f"Precision={metrics['precision']:.3f} "
        f"Recall={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} "
        f"AUC={metrics['auc']:.3f} "
        f"ZoneAcc={sum(zone_acc)/len(zone_acc):.3f} "
        f"ZoneTop3={sum(zone_top3)/len(zone_top3):.3f}"
    )

    print(
        "Explainability | "
        f"Face={mean_weights[0]:.3f}, "
        f"Clothing={mean_weights[1]:.3f}, "
        f"Text={mean_weights[2]:.3f}, "
        f"Geo={mean_weights[3]:.3f}, "
        f"Graph={mean_weights[4]:.3f}"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running RescueNet-X on:", device)

    dataset = SyntheticRescueBenchmark(n=2200)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = RescueNetX(
        vocab_size=1000,
        geo_dim=10,
        graph_node_dim=32,
        num_zones=8
    ).to(device)

    match_loss = nn.BCEWithLogitsLoss()
    zone_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 10

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = to_device(batch, device)
            out = model(batch)

            loss_match = match_loss(out["match_logit"], batch["match_label"])
            loss_zone = zone_loss(out["zone_logits"], batch["zone_label"])

            loss = loss_match + 0.6 * loss_zone

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"Epoch {epoch} | Train Loss={total/len(train_loader):.4f}")
        evaluate(model, val_loader, device)

    torch.save(model.state_dict(), "rescuenet_x_model.pt")
    print("Saved model: rescuenet_x_model.pt")


if __name__ == "__main__":
    main()
