import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import os

from models import GCN, GraphSAGE, GAT


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = (pred[mask] == data.y[mask]).float().mean().item()
        accs.append(acc)
    return accs


def main(model_name, dataset_name, epochs, device):
    # Dataset
    dataset = Planetoid(root="data/Planetoid", name=dataset_name, transform=NormalizeFeatures())
    data = dataset[0]

    # Model selection
    if model_name == "gcn":
        model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
    elif model_name == "sage":
        model = GraphSAGE(dataset.num_features, 16, dataset.num_classes).to(device)
    elif model_name == "gat":
        model = GAT(dataset.num_features, 8, dataset.num_classes).to(device)
    else:
        raise ValueError("Model must be one of: gcn, sage, gat")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Logs
    val_accs, test_accs = [], []

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = evaluate(model, data)

        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss {loss:.3f} | "
                  f"Train {train_acc:.3f} | Val {val_acc:.3f} | Test {test_acc:.3f}")

    # Save plot
    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), val_accs, label="Val Acc")
    plt.plot(range(1, epochs + 1), test_accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name.upper()} on {dataset_name}")
    plt.legend()
    plt.savefig(f"results/{model_name}_{dataset_name}_curve.png")
    print(f"Plot saved to results/{model_name}_{dataset_name}_curve.png")


if __name__ == "__main__":
    # Ask user interactively
    model_name = input("Which model do you want to run? [gcn/sage/gat]: ").strip().lower()
    dataset_name = input("Which dataset do you want to use? [Cora/Citeseer]: ").strip().capitalize()
    epochs = input("How many epochs do you want to train? [default=200]: ").strip()
    device = input("Which device to use? [cpu/cuda, default=cpu]: ").strip().lower()

    # Defaults
    if epochs == "":
        epochs = 200
    else:
        epochs = int(epochs)

    if device == "":
        device = "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available on this system. Using CPU instead.")
        device = "cpu"

    main(model_name, dataset_name, epochs, device)
