import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import csv
import math
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.ou_model import OUSurrogate
from src.ou_data import ou_euler_step


def format_sci(value: float, digits: int = 4) -> str:
    if value == 0:
        return f"{0:.{digits}f}*e^0"

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)

    return f"{mantissa:.{digits}f}*e^{exponent}"


def find_latest_run(outputs_dir: str = "outputs") -> str:
    if not os.path.exists(outputs_dir):
        raise FileNotFoundError("Папка outputs/ не знайдена.")

    runs = [
        os.path.join(outputs_dir, d)
        for d in os.listdir(outputs_dir)
        if d.startswith("ou_run_")
    ]

    if not runs:
        raise FileNotFoundError(
            "Не знайдено жодної папки ou_run_ у outputs/"
        )

    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]


def load_checkpoint(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_model(checkpoint_path: str, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]

    model_cfg = config["model"]

    model = OUSurrogate(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        n_layers=int(model_cfg["n_layers"]),
        activation=str(model_cfg["activation"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def predict_delta(model, device, x, theta, mu, sigma, dt, xi):
    inputs = np.array(
        [
            [
                x,
                theta,
                mu,
                sigma,
                dt,
                xi,
            ]
        ],
        dtype=np.float32,
    )

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(inputs_tensor).cpu().numpy()[0]

    return float(pred[0])


def rollout_true_and_model(model, config, device):
    eval_cfg = config["eval"]
    params = eval_cfg["rollout_params"]

    dt = float(eval_cfg["rollout_dt"])
    steps = int(eval_cfg["rollout_steps"])

    theta = float(params["theta"])
    mu = float(params["mu"])
    sigma = float(params["sigma"])
    x_true = float(params["x0"])
    x_model = float(params["x0"])

    rng = np.random.default_rng(int(config.get("seed", 42)))

    times = [0.0]

    true_values = [x_true]
    model_values = [x_model]
    abs_errors = [0.0]

    for step in range(steps):
        xi = rng.normal()

        x_true_next = ou_euler_step(
            x=x_true,
            theta=theta,
            mu=mu,
            sigma=sigma,
            dt=dt,
            xi=xi,
        )

        pred_delta = predict_delta(
            model=model,
            device=device,
            x=x_model,
            theta=theta,
            mu=mu,
            sigma=sigma,
            dt=dt,
            xi=xi,
        )

        x_model_next = x_model + pred_delta

        x_true = x_true_next
        x_model = x_model_next

        current_time = (step + 1) * dt

        times.append(current_time)
        true_values.append(x_true)
        model_values.append(x_model)
        abs_errors.append(abs(x_true - x_model))

    return {
        "times": np.array(times),
        "true": np.array(true_values),
        "model": np.array(model_values),
        "abs_error": np.array(abs_errors),
    }


def monte_carlo_rollout(model, config, device):
    eval_cfg = config["eval"]
    params = eval_cfg["rollout_params"]

    dt = float(eval_cfg["rollout_dt"])
    steps = int(eval_cfg["rollout_steps"])
    n_paths = int(eval_cfg.get("monte_carlo_paths", 300))

    theta = float(params["theta"])
    mu = float(params["mu"])
    sigma = float(params["sigma"])
    x0 = float(params["x0"])

    rng = np.random.default_rng(int(config.get("seed", 42)))

    times = np.arange(steps + 1) * dt

    true_paths = np.zeros((n_paths, steps + 1), dtype=np.float64)
    model_paths = np.zeros((n_paths, steps + 1), dtype=np.float64)

    true_paths[:, 0] = x0
    model_paths[:, 0] = x0

    for path_idx in range(n_paths):
        x_true = x0
        x_model = x0

        for step in range(steps):
            xi = rng.normal()

            x_true_next = ou_euler_step(
                x=x_true,
                theta=theta,
                mu=mu,
                sigma=sigma,
                dt=dt,
                xi=xi,
            )

            pred_delta = predict_delta(
                model=model,
                device=device,
                x=x_model,
                theta=theta,
                mu=mu,
                sigma=sigma,
                dt=dt,
                xi=xi,
            )

            x_model_next = x_model + pred_delta

            x_true = x_true_next
            x_model = x_model_next

            true_paths[path_idx, step + 1] = x_true
            model_paths[path_idx, step + 1] = x_model

    return {
        "times": times,
        "true_paths": true_paths,
        "model_paths": model_paths,
    }


def read_train_log(log_path: str):
    epochs = []
    train_losses = []
    val_losses = []

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    return np.array(epochs), np.array(train_losses), np.array(val_losses)


def plot_training_dynamics(run_dir: str):
    log_path = os.path.join(run_dir, "train_log.csv")

    if not os.path.exists(log_path):
        print("train_log.csv не знайдено, графік динаміки навчання пропущено.")
        return

    epochs, train_losses, val_losses = read_train_log(log_path)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Навчальна похибка")
    plt.plot(epochs, val_losses, label="Валідаційна похибка")
    plt.yscale("log")
    plt.xlabel("Епоха")
    plt.ylabel("MSE loss, логарифмічна шкала")
    plt.title("Динаміка навчання моделі для процесу Орнштейна-Уленбека")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(run_dir, "динаміка_навчання_орнштейн_уленбек.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_rollout(rollout_data: dict, run_dir: str):
    times = rollout_data["times"]

    plt.figure(figsize=(10, 5))
    plt.plot(times, rollout_data["true"], label="Істинна траєкторія")
    plt.plot(times, rollout_data["model"], label="Модельна траєкторія", linestyle="--")
    plt.xlabel("Час")
    plt.ylabel("Значення X(t)")
    plt.title("Порівняння істинної та модельної траєкторії процесу Орнштейна-Уленбека")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(run_dir, "траєкторія_орнштейн_уленбек.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_abs_error(rollout_data: dict, run_dir: str):
    times = rollout_data["times"]

    plt.figure(figsize=(10, 5))
    plt.plot(times, rollout_data["abs_error"], label="Абсолютна похибка")
    plt.xlabel("Час")
    plt.ylabel("Абсолютна похибка")
    plt.title("Абсолютна похибка моделювання процесу Орнштейна-Уленбека")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(run_dir, "абсолютна_похибка_орнштейн_уленбек.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_monte_carlo_mean_std(mc_data: dict, run_dir: str):
    times = mc_data["times"]
    true_paths = mc_data["true_paths"]
    model_paths = mc_data["model_paths"]

    true_mean = true_paths.mean(axis=0)
    true_std = true_paths.std(axis=0)

    model_mean = model_paths.mean(axis=0)
    model_std = model_paths.std(axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(times, true_mean, label="Істинне середнє")
    plt.plot(times, model_mean, label="Модельне середнє", linestyle="--")

    plt.fill_between(
        times,
        true_mean - true_std,
        true_mean + true_std,
        alpha=0.2,
        label="Істинне середнє ± std",
    )

    plt.fill_between(
        times,
        model_mean - model_std,
        model_mean + model_std,
        alpha=0.2,
        label="Модельне середнє ± std",
    )

    plt.xlabel("Час")
    plt.ylabel("Значення X(t)")
    plt.title("Монте-Карло моделювання процесу Орнштейна-Уленбека")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(run_dir, "монте_карло_середнє_std_орнштейн_уленбек.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_terminal_distribution(mc_data: dict, run_dir: str):
    true_terminal = mc_data["true_paths"][:, -1]
    model_terminal = mc_data["model_paths"][:, -1]

    plt.figure(figsize=(10, 5))
    plt.hist(true_terminal, bins=30, alpha=0.6, label="Істинні кінцеві стани")
    plt.hist(model_terminal, bins=30, alpha=0.6, label="Модельні кінцеві стани")
    plt.xlabel("Значення X(T)")
    plt.ylabel("Частота")
    plt.title("Розподіл кінцевих станів процесу Орнштейна-Уленбека")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(run_dir, "монте_карло_кінцеві_стани_орнштейн_уленбек.png")
    plt.savefig(path, dpi=200)
    plt.close()


def save_eval_metrics(rollout_data: dict, mc_data: dict, run_dir: str):
    abs_error = rollout_data["abs_error"]

    rollout_mae = float(np.mean(abs_error))
    rollout_max_abs_error = float(np.max(abs_error))

    true_terminal = mc_data["true_paths"][:, -1]
    model_terminal = mc_data["model_paths"][:, -1]

    true_mean_terminal = float(np.mean(true_terminal))
    model_mean_terminal = float(np.mean(model_terminal))

    true_std_terminal = float(np.std(true_terminal))
    model_std_terminal = float(np.std(model_terminal))

    metrics_path = os.path.join(run_dir, "eval_metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Evaluation для процесу Орнштейна-Уленбека\n")
        f.write("=" * 70 + "\n")
        f.write(f"Rollout MAE: {rollout_mae:.10f}\n")
        f.write(f"Rollout max absolute error: {rollout_max_abs_error:.10f}\n")
        f.write("\n")
        f.write("Monte Carlo terminal statistics\n")
        f.write(f"True mean X(T): {true_mean_terminal:.10f}\n")
        f.write(f"Model mean X(T): {model_mean_terminal:.10f}\n")
        f.write(f"True std X(T): {true_std_terminal:.10f}\n")
        f.write(f"Model std X(T): {model_std_terminal:.10f}\n")

    print("Evaluation-метрики:")
    print(f"Rollout MAE: {rollout_mae:.10f}")
    print(f"Rollout max absolute error: {rollout_max_abs_error:.10f}")
    print(f"True mean X(T): {true_mean_terminal:.10f}")
    print(f"Model mean X(T): {model_mean_terminal:.10f}")
    print(f"True std X(T): {true_std_terminal:.10f}")
    print(f"Model std X(T): {model_std_terminal:.10f}")
    print(f"Метрики збережено: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Оцінювання моделі для процесу Орнштейна-Уленбека"
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Папка запуску. Якщо не задано, буде використано останню ou_run_*",
    )

    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run()
    checkpoint_path = os.path.join(run_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Не знайдено модель: {checkpoint_path}")

    print("Починаємо evaluation для процесу Орнштейна-Уленбека...")
    print(f"Папка запуску: {run_dir}")
    print(f"Модель: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")

    model, config = load_model(checkpoint_path, device)

    plot_training_dynamics(run_dir)

    rollout_data = rollout_true_and_model(
        model=model,
        config=config,
        device=device,
    )

    mc_data = monte_carlo_rollout(
        model=model,
        config=config,
        device=device,
    )

    plot_rollout(rollout_data, run_dir)
    plot_abs_error(rollout_data, run_dir)
    plot_monte_carlo_mean_std(mc_data, run_dir)
    plot_terminal_distribution(mc_data, run_dir)

    save_eval_metrics(
        rollout_data=rollout_data,
        mc_data=mc_data,
        run_dir=run_dir,
    )

    print("Evaluation завершено.")
    print("Графіки збережено у папці запуску.")


if __name__ == "__main__":
    main()