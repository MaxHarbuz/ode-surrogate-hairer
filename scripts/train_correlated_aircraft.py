import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import csv
import math
import time
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.correlated_aircraft_data import (
    load_correlated_aircraft_npz,
    CorrelatedAircraftDataset,
    make_correlated_aircraft_splits,
    make_correlated_aircraft_loaders,
)

from src.correlated_aircraft_model import CorrelatedAircraftSurrogate


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_run_dir(base_dir: str = "outputs") -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"correlated_aircraft_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(config: dict, run_dir: str):
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def format_sci(value: float, digits: int = 4) -> str:
    """
    Форматування числа у вигляді:
    7.1111*e^-6
    """
    if value == 0:
        return f"{0:.{digits}f}*e^0"

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)

    return f"{mantissa:.{digits}f}*e^{exponent}"


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    total_count = 0

    progress_bar = tqdm(
        loader,
        desc=f"Епоха {epoch:03d}",
        leave=False,
        ncols=120,
    )

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        current_avg_loss = total_loss / total_count

        progress_bar.set_postfix(
            {
                "train_loss": format_sci(current_avg_loss)
            }
        )

    return total_loss / total_count


def evaluate_loss(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / total_count


def save_train_log(log_path: str, rows: list):
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Навчання нейромережевої моделі для корельованої стохастичної системи літака"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/correlated_aircraft.yaml",
        help="Шлях до конфігураційного файлу",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    dataset_path = config["paths"]["dataset_path"]

    print("Починаємо навчання моделі для корельованої системи літака...")
    print(f"Файл конфігурації: {args.config}")
    print(f"Датасет: {dataset_path}")

    inputs, targets = load_correlated_aircraft_npz(dataset_path)

    dataset = CorrelatedAircraftDataset(inputs, targets)

    train_ds, val_ds, test_ds = make_correlated_aircraft_splits(
        dataset=dataset,
        config=config,
    )

    train_loader, val_loader, test_loader = make_correlated_aircraft_loaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій для навчання: {device}")

    model_cfg = config["model"]

    model = CorrelatedAircraftSurrogate(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        n_layers=int(model_cfg["n_layers"]),
        activation=str(model_cfg["activation"]),
    ).to(device)

    train_cfg = config["train"]

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    min_delta = float(train_cfg["early_stopping_min_delta"])

    run_dir = create_run_dir()
    save_config(config, run_dir)

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    train_log = []

    best_model_path = os.path.join(run_dir, "best_model.pt")
    log_path = os.path.join(run_dir, "train_log.csv")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        improved = False

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            improved = True

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                best_model_path,
            )
        else:
            bad_epochs += 1

        train_log.append([epoch, train_loss, val_loss, best_val_loss])

        improvement_mark = " ← best" if improved else ""

        print(
            f"Епоха {epoch:03d} | "
            f"train_loss = {format_sci(train_loss)} | "
            f"val_loss = {format_sci(val_loss)} | "
            f"best_val_loss = {format_sci(best_val_loss)}"
            f"{improvement_mark}"
        )

        if bad_epochs >= patience:
            print(
                f"Early stopping на епосі {epoch}. "
                f"Найкраща епоха: {best_epoch}, "
                f"найкращий val_loss: {format_sci(best_val_loss)}"
            )
            break

    save_train_log(log_path, train_log)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate_loss(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics_path = os.path.join(run_dir, "test_metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Результати тестування моделі для корельованої системи літака\n")
        f.write("=" * 75 + "\n")
        f.write(f"Найкраща епоха: {best_epoch}\n")
        f.write(f"Найкращий validation loss: {format_sci(best_val_loss)}\n")
        f.write(f"Test MSE loss: {format_sci(test_loss)}\n")

    print("Навчання завершено.")
    print(f"Папка запуску: {run_dir}")
    print(f"Найкраща модель: {best_model_path}")
    print(f"Лог навчання: {log_path}")
    print(f"Метрики тестування: {metrics_path}")
    print(f"Test MSE loss: {format_sci(test_loss)}")


if __name__ == "__main__":
    main()