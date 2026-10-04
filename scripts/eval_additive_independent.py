import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import os
import csv
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.additive_independent_model import AdditiveIndependentSurrogate
from src.additive_independent_data import additive_independent_euler_step


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_run(outputs_dir: str = "outputs") -> str:
    runs = [
        os.path.join(outputs_dir, d)
        for d in os.listdir(outputs_dir)
        if d.startswith("additive_independent_run_")
    ]

    if not runs:
        raise FileNotFoundError(
            "Не знайдено жодної папки additive_independent_run_ у outputs/"
        )

    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]


def load_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model_cfg = config["model"]

    model = AdditiveIndependentSurrogate(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        n_layers=int(model_cfg["n_layers"]),
        activation=str(model_cfg["activation"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def predict_delta(
    model,
    device,
    x1,
    x2,
    a1,
    b1,
    sigma1,
    a2,
    b2,
    sigma2,
    dt,
    xi1,
    xi2,
):
    inputs = np.array(
        [
            [
                x1,
                x2,
                a1,
                b1,
                sigma1,
                a2,
                b2,
                sigma2,
                dt,
                xi1,
                xi2,
            ]
        ],
        dtype=np.float32,
    )

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(inputs_tensor).cpu().numpy()[0]

    return pred[0], pred[1]


def rollout_true_and_model(model, config, device):
    eval_cfg = config["eval"]
    params = eval_cfg["rollout_params"]

    dt = float(eval_cfg["rollout_dt"])
    steps = int(eval_cfg["rollout_steps"])

    x1_true = float(params["x1_0"])
    x2_true = float(params["x2_0"])

    x1_model = float(params["x1_0"])
    x2_model = float(params["x2_0"])

    a1 = float(params["a1"])
    b1 = float(params["b1"])
    sigma1 = float(params["sigma1"])

    a2 = float(params["a2"])
    b2 = float(params["b2"])
    sigma2 = float(params["sigma2"])

    rng = np.random.default_rng(int(config.get("seed", 42)))

    times = [0.0]

    x1_true_values = [x1_true]
    x2_true_values = [x2_true]

    x1_model_values = [x1_model]
    x2_model_values = [x2_model]

    abs_error_x1 = [0.0]
    abs_error_x2 = [0.0]

    for step in range(steps):
        xi1 = rng.normal()
        xi2 = rng.normal()

        x1_true_next, x2_true_next = additive_independent_euler_step(
            x1=x1_true,
            x2=x2_true,
            a1=a1,
            b1=b1,
            sigma1=sigma1,
            a2=a2,
            b2=b2,
            sigma2=sigma2,
            dt=dt,
            xi1=xi1,
            xi2=xi2,
        )

        pred_delta_x1, pred_delta_x2 = predict_delta(
            model=model,
            device=device,
            x1=x1_model,
            x2=x2_model,
            a1=a1,
            b1=b1,
            sigma1=sigma1,
            a2=a2,
            b2=b2,
            sigma2=sigma2,
            dt=dt,
            xi1=xi1,
            xi2=xi2,
        )

        x1_model_next = x1_model + pred_delta_x1
        x2_model_next = x2_model + pred_delta_x2

        x1_true = x1_true_next
        x2_true = x2_true_next

        x1_model = x1_model_next
        x2_model = x2_model_next

        current_time = (step + 1) * dt
        times.append(current_time)

        x1_true_values.append(x1_true)
        x2_true_values.append(x2_true)

        x1_model_values.append(x1_model)
        x2_model_values.append(x2_model)

        abs_error_x1.append(abs(x1_true - x1_model))
        abs_error_x2.append(abs(x2_true - x2_model))

    return {
        "times": np.array(times),
        "x1_true": np.array(x1_true_values),
        "x2_true": np.array(x2_true_values),
        "x1_model": np.array(x1_model_values),
        "x2_model": np.array(x2_model_values),
        "abs_error_x1": np.array(abs_error_x1),
        "abs_error_x2": np.array(abs_error_x2),
    }
def monte_carlo_rollout(model, config, device):
    eval_cfg = config["eval"]
    params = eval_cfg["rollout_params"]

    dt = float(eval_cfg["rollout_dt"])
    steps = int(eval_cfg["rollout_steps"])
    n_paths = int(eval_cfg.get("monte_carlo_paths", 200))

    x1_0 = float(params["x1_0"])
    x2_0 = float(params["x2_0"])

    a1 = float(params["a1"])
    b1 = float(params["b1"])
    sigma1 = float(params["sigma1"])

    a2 = float(params["a2"])
    b2 = float(params["b2"])
    sigma2 = float(params["sigma2"])

    rng = np.random.default_rng(int(config.get("seed", 42)))

    times = np.arange(steps + 1) * dt

    true_x1 = np.zeros((n_paths, steps + 1), dtype=np.float64)
    true_x2 = np.zeros((n_paths, steps + 1), dtype=np.float64)
    model_x1 = np.zeros((n_paths, steps + 1), dtype=np.float64)
    model_x2 = np.zeros((n_paths, steps + 1), dtype=np.float64)

    true_x1[:, 0] = x1_0
    true_x2[:, 0] = x2_0
    model_x1[:, 0] = x1_0
    model_x2[:, 0] = x2_0

    for path_idx in range(n_paths):
        x1_true = x1_0
        x2_true = x2_0
        x1_model = x1_0
        x2_model = x2_0

        for step in range(steps):
            xi1 = rng.normal()
            xi2 = rng.normal()

            x1_true_next, x2_true_next = additive_independent_euler_step(
                x1=x1_true,
                x2=x2_true,
                a1=a1,
                b1=b1,
                sigma1=sigma1,
                a2=a2,
                b2=b2,
                sigma2=sigma2,
                dt=dt,
                xi1=xi1,
                xi2=xi2,
            )

            pred_delta_x1, pred_delta_x2 = predict_delta(
                model=model,
                device=device,
                x1=x1_model,
                x2=x2_model,
                a1=a1,
                b1=b1,
                sigma1=sigma1,
                a2=a2,
                b2=b2,
                sigma2=sigma2,
                dt=dt,
                xi1=xi1,
                xi2=xi2,
            )

            x1_model_next = x1_model + pred_delta_x1
            x2_model_next = x2_model + pred_delta_x2

            x1_true = x1_true_next
            x2_true = x2_true_next
            x1_model = x1_model_next
            x2_model = x2_model_next

            true_x1[path_idx, step + 1] = x1_true
            true_x2[path_idx, step + 1] = x2_true
            model_x1[path_idx, step + 1] = x1_model
            model_x2[path_idx, step + 1] = x2_model

    return {
        "times": times,
        "true_x1": true_x1,
        "true_x2": true_x2,
        "model_x1": model_x1,
        "model_x2": model_x2,
    }

def plot_mc_mean_std_x1(mc_results, run_dir):
    times = mc_results["times"]

    true_mean = np.mean(mc_results["true_x1"], axis=0)
    true_std = np.std(mc_results["true_x1"], axis=0)

    model_mean = np.mean(mc_results["model_x1"], axis=0)
    model_std = np.std(mc_results["model_x1"], axis=0)

    plt.figure(figsize=(10, 6))

    plt.plot(times, true_mean, label="Істинне середнє X₁")
    plt.fill_between(
        times,
        true_mean - true_std,
        true_mean + true_std,
        alpha=0.25,
        label="Істинне середнє ± std X₁",
    )

    plt.plot(times, model_mean, linestyle="--", label="Прогнозоване середнє X₁")
    plt.fill_between(
        times,
        model_mean - model_std,
        model_mean + model_std,
        alpha=0.25,
        label="Прогнозоване середнє ± std X₁",
    )

    plt.xlabel("Час")
    plt.ylabel("Стан X₁")
    plt.title("Monte Carlo: середнє значення та стандартне відхилення для X₁")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(
        run_dir,
        "монте_карло_середнє_std_x1_адитивна_незалежна.png",
    )
    plt.savefig(path, dpi=300)
    plt.close()

def plot_mc_mean_std_x2(mc_results, run_dir):
    times = mc_results["times"]

    true_mean = np.mean(mc_results["true_x2"], axis=0)
    true_std = np.std(mc_results["true_x2"], axis=0)

    model_mean = np.mean(mc_results["model_x2"], axis=0)
    model_std = np.std(mc_results["model_x2"], axis=0)

    plt.figure(figsize=(10, 6))

    plt.plot(times, true_mean, label="Істинне середнє X₂")
    plt.fill_between(
        times,
        true_mean - true_std,
        true_mean + true_std,
        alpha=0.25,
        label="Істинне середнє ± std X₂",
    )

    plt.plot(times, model_mean, linestyle="--", label="Прогнозоване середнє X₂")
    plt.fill_between(
        times,
        model_mean - model_std,
        model_mean + model_std,
        alpha=0.25,
        label="Прогнозоване середнє ± std X₂",
    )

    plt.xlabel("Час")
    plt.ylabel("Стан X₂")
    plt.title("Monte Carlo: середнє значення та стандартне відхилення для X₂")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(
        run_dir,
        "монте_карло_середнє_std_x2_адитивна_незалежна.png",
    )
    plt.savefig(path, dpi=300)
    plt.close()

def plot_mc_terminal_phase_space(mc_results, run_dir):
    true_x1_terminal = mc_results["true_x1"][:, -1]
    true_x2_terminal = mc_results["true_x2"][:, -1]

    model_x1_terminal = mc_results["model_x1"][:, -1]
    model_x2_terminal = mc_results["model_x2"][:, -1]

    plt.figure(figsize=(8, 8))

    plt.scatter(
        true_x1_terminal,
        true_x2_terminal,
        alpha=0.6,
        s=18,
        label="Істинні кінцеві стани",
    )

    plt.scatter(
        model_x1_terminal,
        model_x2_terminal,
        alpha=0.6,
        s=18,
        label="Прогнозовані кінцеві стани",
        marker="x",
    )

    plt.xlabel("X₁(T)")
    plt.ylabel("X₂(T)")
    plt.title("Monte Carlo: розподіл кінцевих станів у фазовому просторі")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(
        run_dir,
        "монте_карло_кінцеві_стани_фазовий_простір_адитивна_незалежна.png",
    )
    plt.savefig(path, dpi=300)
    plt.close()

def plot_rollout(results, run_dir):
    times = results["times"]

    plt.figure(figsize=(10, 6))
    plt.plot(times, results["x1_true"], label="Істинна траєкторія X₁")
    plt.plot(times, results["x1_model"], label="Прогноз моделі X₁", linestyle="--")
    plt.xlabel("Час")
    plt.ylabel("Стан X₁")
    plt.title("Порівняння істинної та прогнозованої траєкторії X₁")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(run_dir, "траєкторія_x1_адитивна_незалежна.png")
    plt.savefig(path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(times, results["x2_true"], label="Істинна траєкторія X₂")
    plt.plot(times, results["x2_model"], label="Прогноз моделі X₂", linestyle="--")
    plt.xlabel("Час")
    plt.ylabel("Стан X₂")
    plt.title("Порівняння істинної та прогнозованої траєкторії X₂")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(run_dir, "траєкторія_x2_адитивна_незалежна.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_errors(results, run_dir):
    times = results["times"]

    plt.figure(figsize=(10, 6))
    plt.plot(times, results["abs_error_x1"], label="Абсолютна похибка X₁")
    plt.plot(times, results["abs_error_x2"], label="Абсолютна похибка X₂")
    plt.xlabel("Час")
    plt.ylabel("Абсолютна похибка")
    plt.title("Абсолютна похибка прогнозу для адитивної незалежної системи")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(run_dir, "абсолютна_похибка_адитивна_незалежна.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_phase_trajectory(results, run_dir):
    plt.figure(figsize=(8, 8))
    plt.plot(
        results["x1_true"],
        results["x2_true"],
        label="Істинна фазова траєкторія",
    )
    plt.plot(
        results["x1_model"],
        results["x2_model"],
        label="Фазова траєкторія моделі",
        linestyle="--",
    )
    plt.xlabel("X₁")
    plt.ylabel("X₂")
    plt.title("Фазова траєкторія адитивної незалежної системи")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(run_dir, "фазова_траєкторія_адитивна_незалежна.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_train_log(run_dir):
    import matplotlib.ticker as mticker

    log_path = os.path.join(run_dir, "train_log.csv")

    if not os.path.exists(log_path):
        print("Файл train_log.csv не знайдено, графік навчання не побудовано.")
        return

    epochs = []
    train_losses = []
    val_losses = []

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Навчальна вибірка")
    plt.plot(epochs, val_losses, label="Валідаційна вибірка")

    plt.xlabel("Епоха")
    plt.ylabel("MSE loss")
    plt.title("Динаміка навчання моделі для адитивної незалежної системи")

    # Логарифмічний масштаб по осі Y
    plt.yscale("log")

    # Фіксовані тики по степенях 10
    plt.yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    # Формат підписів у вигляді 10^-7, 10^-6 і т.д.
    plt.gca().yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    path = os.path.join(run_dir, "динаміка_навчання_адитивна_незалежна.png")
    plt.savefig(path, dpi=300)
    plt.close()

def save_rollout_metrics(results, run_dir):
    mean_abs_error_x1 = float(np.mean(results["abs_error_x1"]))
    mean_abs_error_x2 = float(np.mean(results["abs_error_x2"]))

    max_abs_error_x1 = float(np.max(results["abs_error_x1"]))
    max_abs_error_x2 = float(np.max(results["abs_error_x2"]))

    metrics_path = os.path.join(run_dir, "rollout_metrics_additive_independent.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Метрики rollout-оцінювання для адитивної незалежної системи\n")
        f.write("=" * 75 + "\n")
        f.write(f"Середня абсолютна похибка X1: {mean_abs_error_x1:.10f}\n")
        f.write(f"Середня абсолютна похибка X2: {mean_abs_error_x2:.10f}\n")
        f.write(f"Максимальна абсолютна похибка X1: {max_abs_error_x1:.10f}\n")
        f.write(f"Максимальна абсолютна похибка X2: {max_abs_error_x2:.10f}\n")

    print(f"Метрики rollout збережено у файл: {metrics_path}")

def save_mc_metrics(mc_results, run_dir):
    true_mean_x1_final = float(np.mean(mc_results["true_x1"][:, -1]))
    true_std_x1_final = float(np.std(mc_results["true_x1"][:, -1]))

    model_mean_x1_final = float(np.mean(mc_results["model_x1"][:, -1]))
    model_std_x1_final = float(np.std(mc_results["model_x1"][:, -1]))

    true_mean_x2_final = float(np.mean(mc_results["true_x2"][:, -1]))
    true_std_x2_final = float(np.std(mc_results["true_x2"][:, -1]))

    model_mean_x2_final = float(np.mean(mc_results["model_x2"][:, -1]))
    model_std_x2_final = float(np.std(mc_results["model_x2"][:, -1]))

    metrics_path = os.path.join(run_dir, "mc_metrics_additive_independent.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Monte Carlo-метрики для адитивної незалежної системи\n")
        f.write("=" * 75 + "\n")
        f.write(f"Істинне середнє X1(T): {true_mean_x1_final:.10f}\n")
        f.write(f"Істинне std X1(T): {true_std_x1_final:.10f}\n")
        f.write(f"Прогнозоване середнє X1(T): {model_mean_x1_final:.10f}\n")
        f.write(f"Прогнозоване std X1(T): {model_std_x1_final:.10f}\n")
        f.write("\n")
        f.write(f"Істинне середнє X2(T): {true_mean_x2_final:.10f}\n")
        f.write(f"Істинне std X2(T): {true_std_x2_final:.10f}\n")
        f.write(f"Прогнозоване середнє X2(T): {model_mean_x2_final:.10f}\n")
        f.write(f"Прогнозоване std X2(T): {model_std_x2_final:.10f}\n")

    print(f"Monte Carlo-метрики збережено у файл: {metrics_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Оцінювання моделі для адитивної незалежної стохастичної системи"
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Папка запуску. Якщо не вказано, буде використано останній additive_independent_run.",
    )

    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_run()
    else:
        run_dir = args.run_dir

    checkpoint_path = os.path.join(run_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Не знайдено модель: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Починаємо оцінювання моделі для адитивної незалежної системи...")
    print(f"Папка запуску: {run_dir}")
    print(f"Модель: {checkpoint_path}")
    print(f"Пристрій: {device}")

    model, config = load_model(checkpoint_path, device)

    results = rollout_true_and_model(
        model=model,
        config=config,
        device=device,
    )

    mc_results = monte_carlo_rollout(
        model=model,
        config=config,
        device=device,
    )
    plot_train_log(run_dir)
    plot_rollout(results, run_dir)
    plot_errors(results, run_dir)
    plot_phase_trajectory(results, run_dir)
    save_rollout_metrics(results, run_dir)

    plot_mc_mean_std_x1(mc_results, run_dir)
    plot_mc_mean_std_x2(mc_results, run_dir)
    plot_mc_terminal_phase_space(mc_results, run_dir)
    save_mc_metrics(mc_results, run_dir)

    print("Оцінювання завершено.")
    print("Побудовано графіки:")
    print("- динаміка_навчання_адитивна_незалежна.png")
    print("- траєкторія_x1_адитивна_незалежна.png")
    print("- траєкторія_x2_адитивна_незалежна.png")
    print("- абсолютна_похибка_адитивна_незалежна.png")
    print("- фазова_траєкторія_адитивна_незалежна.png")
    print("- монте_карло_середнє_std_x1_адитивна_незалежна.png")
    print("- монте_карло_середнє_std_x2_адитивна_незалежна.png")
    print("- монте_карло_кінцеві_стани_фазовий_простір_адитивна_незалежна.png")


if __name__ == "__main__":
    main()