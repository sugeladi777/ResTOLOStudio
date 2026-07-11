from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from ml.molecule_preprocessing import DEFAULT_PREPROCESSING, image_to_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class FolderDataset(Dataset):
    def __init__(self, root: str, class_names: list[str]):
        self.samples: list[tuple[Path, int]] = []
        for class_index, class_name in enumerate(class_names):
            class_dir = Path(root) / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(DEFAULT_PREPROCESSING.target_size, Image.Resampling.BICUBIC)
            tensor = image_to_tensor(image)
        return tensor, target


def _checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    return checkpoint


def _build_model(pretrained_model: str, class_count: int) -> nn.Module:
    checkpoint = torch.load(pretrained_model, map_location="cpu", weights_only=False)
    state_dict = _checkpoint_state(checkpoint)
    if "layer3.22.conv1.weight" in state_dict:
        model = models.resnet101(weights=None)
        backbone_name = "resnet101"
    elif "layer3.5.conv1.weight" in state_dict:
        layer_weight = state_dict["layer3.5.conv1.weight"]
        if tuple(layer_weight.shape[-2:]) == (1, 1):
            model = models.resnet50(weights=None)
            backbone_name = "resnet50"
        else:
            model = models.resnet34(weights=None)
            backbone_name = "resnet34"
    elif "layer3.1.conv1.weight" in state_dict:
        model = models.resnet18(weights=None)
        backbone_name = "resnet18"
    else:
        model = models.resnet50(weights=None)
        backbone_name = "resnet50"

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, class_count)
    backbone_state = {key: value for key, value in state_dict.items() if key not in {"fc.weight", "fc.bias"}}
    model.load_state_dict(backbone_state, strict=False)
    model.backbone_name = backbone_name
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")
    return model


def _macro_f1(predictions: torch.Tensor, targets: torch.Tensor, class_count: int) -> tuple[float, dict[str, dict]]:
    metrics: dict[str, dict] = {}
    values: list[float] = []
    for class_index in range(class_count):
        support = int((targets == class_index).sum())
        if support == 0:
            continue
        true_positive = int(((predictions == class_index) & (targets == class_index)).sum())
        false_positive = int(((predictions == class_index) & (targets != class_index)).sum())
        false_negative = int(((predictions != class_index) & (targets == class_index)).sum())
        f1 = 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)
        recall = true_positive / max(1, true_positive + false_negative)
        metrics[str(class_index)] = {"support": support, "f1": f1, "recall": recall}
        values.append(f1)
    return (sum(values) / len(values) if values else 0.0), metrics


def _reported_class_metrics(
    per_class: dict[str, dict],
    class_names: list[str],
    class_indices: list[int],
    total_counts: Counter,
) -> dict[str, dict]:
    reported: dict[str, dict] = {}
    for local_index, (class_name, original_index) in enumerate(zip(class_names, class_indices)):
        values = per_class.get(str(local_index), {})
        validation_support = int(values.get("support", 0))
        sample_count = int(total_counts.get(local_index, 0))
        reported[str(original_index)] = {
            "class_name": class_name,
            "sample_count": sample_count,
            "validation_support": validation_support,
            "validation_status": "verifiable" if sample_count > 1 and validation_support > 0 else "insufficient",
            "recall": values.get("recall"),
            "f1": values.get("f1"),
        }
    return reported


def _evaluate(model, loader, criterion, device, class_count):
    model.eval()
    losses: list[float] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            logits = model(images)
            losses.append(float(criterion(logits, batch_targets).detach().cpu()))
            predictions.append(logits.argmax(1).cpu())
            targets.append(batch_targets.cpu())
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    macro_f1, per_class = _macro_f1(all_predictions, all_targets, class_count)
    accuracy = float((all_predictions == all_targets).float().mean())
    return sum(losses) / max(1, len(losses)), accuracy, macro_f1, per_class


def _extract_features(model, loader, device):
    backbone = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, batch_targets in loader:
            batch_features = backbone(images.to(device, non_blocking=True)).flatten(1)
            features.append(batch_features)
            targets.append(batch_targets.to(device, non_blocking=True))
    return torch.cat(features), torch.cat(targets)


def _evaluate_head(head, features, targets, criterion, class_count):
    with torch.no_grad():
        logits = head(features)
        loss = float(criterion(logits, targets).detach().cpu())
        predictions = logits.argmax(1).cpu()
        cpu_targets = targets.cpu()
    accuracy = float((predictions == cpu_targets).float().mean())
    macro_f1, per_class = _macro_f1(predictions, cpu_targets, class_count)
    return loss, accuracy, macro_f1, per_class


def _class_metadata(saving_path: Path, folder_names: list[str]) -> tuple[list[str], list[int]]:
    classes_path = saving_path / "classes.yaml"
    if not classes_path.exists():
        inferred_indices = [int(name) if name.lstrip("-").isdigit() else index for index, name in enumerate(folder_names)]
        return folder_names, inferred_indices
    payload = yaml.safe_load(classes_path.read_text(encoding="utf-8")) or {}
    names = [str(name) for name in payload.get("names", folder_names)]
    indices = [int(index) for index in payload.get("indices", range(len(names)))]
    return names, indices


def main():
    parser = argparse.ArgumentParser(description="Stable small-dataset ResNet fine-tuning")
    parser.add_argument("--training_path", required=True)
    parser.add_argument("--testing_path", required=True)
    parser.add_argument("--saving_path", required=True)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    saving_path = Path(args.saving_path)
    saving_path.mkdir(parents=True, exist_ok=True)
    folder_names = sorted(path.name for path in Path(args.training_path).iterdir() if path.is_dir())
    if not folder_names:
        raise ValueError("分类训练集没有类别目录")

    class_names, class_indices = _class_metadata(saving_path, folder_names)
    if len(class_names) != len(folder_names):
        raise ValueError(f"类别文件数量 {len(class_names)} 与训练目录数量 {len(folder_names)} 不一致")

    train_dataset = FolderDataset(args.training_path, folder_names)
    val_dataset = FolderDataset(args.testing_path, folder_names)
    if not train_dataset or not val_dataset:
        raise ValueError("分类训练集或验证集为空")

    counts = Counter(target for _, target in train_dataset.samples)
    total_counts = Counter(target for _, target in train_dataset.samples + val_dataset.samples)
    weights = torch.tensor(
        [len(train_dataset) / max(1, len(folder_names) * counts.get(index, 0)) for index in range(len(folder_names))],
        dtype=torch.float32,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 类别统计: {dict(counts)}")
    print("数据增强: 关闭；类别不平衡处理: 加权交叉熵")

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(max(args.batch_size, 32), len(train_dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(max(args.batch_size, 16), len(val_dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    torch.manual_seed(20260711)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260711)
    model = _build_model(args.pretrained_model, len(folder_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    train_features, train_targets = _extract_features(model, train_loader, device)
    val_features, val_targets = _extract_features(model, val_loader, device)
    head = model.fc
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    best_f1 = -1.0
    best_epoch = 0
    best_metrics = {}
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(train_features), train_targets)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())
        head.eval()
        val_loss, accuracy, macro_f1, per_class = _evaluate_head(
            head, val_features, val_targets, criterion, len(folder_names)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )
        print(
            f"Training Loss: {train_loss:.8f} Prediction Error: {val_loss:.8f} Epoch {epoch} "
            f"Accuracy: {accuracy:.4f} Macro-F1: {macro_f1:.4f}",
            flush=True,
        )

        validation_metrics = {
            "val_loss": val_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": _reported_class_metrics(per_class, class_names, class_indices, total_counts),
            "class_counts": {
                str(class_indices[index]): int(total_counts.get(index, 0))
                for index in range(len(folder_names))
            },
        }
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "class_indices": class_indices,
            "preprocessing": DEFAULT_PREPROCESSING.to_dict(),
            "backbone": model.backbone_name,
            "metrics": validation_metrics,
            "validation_metrics": validation_metrics,
            "history": history,
        }
        torch.save(checkpoint, saving_path / "Model_last.saving")
        if macro_f1 > best_f1 + 1e-6:
            best_f1 = macro_f1
            best_epoch = epoch
            best_metrics = dict(checkpoint["metrics"])
            epochs_without_improvement = 0
            torch.save(checkpoint, saving_path / "Model_best.saving")
        else:
            epochs_without_improvement += 1

        patience = max(args.patience, 60)
        if epochs_without_improvement >= patience:
            print(f"早停：连续 {patience} 轮 Macro-F1 未提升，最佳轮次 {best_epoch}")
            break

    all_features = torch.cat((train_features, val_features))
    all_targets = torch.cat((train_targets, val_targets))
    all_counts = torch.bincount(all_targets, minlength=len(folder_names)).float()
    all_weights = torch.tensor(
        [len(all_targets) / max(1, len(folder_names) * float(all_counts[index])) for index in range(len(folder_names))],
        dtype=torch.float32,
        device=device,
    )
    torch.manual_seed(20260711)
    final_head = nn.Linear(model.fc.in_features, len(folder_names)).to(device)
    final_optimizer = torch.optim.AdamW(final_head.parameters(), lr=1e-2, weight_decay=1e-3)
    final_criterion = nn.CrossEntropyLoss(weight=all_weights)
    for _ in range(max(1, best_epoch)):
        final_head.train()
        final_optimizer.zero_grad()
        final_loss = final_criterion(final_head(all_features), all_targets)
        final_loss.backward()
        final_optimizer.step()
    model.fc = final_head
    final_checkpoint = {
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_indices": class_indices,
        "preprocessing": DEFAULT_PREPROCESSING.to_dict(),
        "backbone": model.backbone_name,
        "metrics": best_metrics,
        "validation_metrics": best_metrics,
        "final_training": {
            "samples": len(all_targets),
            "steps": best_epoch,
            "class_counts": {
                str(class_indices[index]): int(all_counts[index])
                for index in range(len(folder_names))
            },
        },
        "history": history,
    }
    torch.save(final_checkpoint, saving_path / "Model_final.saving")
    print(f"最终分类模型已使用全部 {len(all_targets)} 个样本训练 {best_epoch} 步")

    report = {
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "device": str(device),
        "class_names": class_names,
        "class_indices": class_indices,
        "preprocessing": DEFAULT_PREPROCESSING.to_dict(),
        "validation_metrics": best_metrics,
        "history": history,
        "final_model": "Model_final.saving",
        "final_training_samples": len(all_targets),
    }
    (saving_path / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
