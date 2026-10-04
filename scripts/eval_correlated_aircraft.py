import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import csv
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.correlated_aircraft_model import CorrelatedAircraftSurrogate
from src.correlated_aircraft_data import correlated_aircraft_euler_step


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_run(outputs_dir: str = "outputs") -> str:
    runs = [
        os.path.join(outputs_dir, d)
        for d in os.listdir(outputs_dir)
        if d.startswith("correlated_aircraft_run_")
    ]

    if not runs:
        raise FileNotFoundError(
            "Не знайдено жодної папки correlated_aircraft_run_ у outputs/"
        )

    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]


def load_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model_cfg = config["model"]

    model = CorrelatedAircraftSurrogate(
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
    h,
    v,
    a_h,
    b_h,
    c_h,
    sigma_h,
    k_h,
    a_v,
    b_v,
    c_v,
    sigma_v,
    k_v,
    rho,
    dt,
    xi1,
    xi2,
):
    inputs = np.array(
        [
            [
                h,
                v,
                a_h,
                b_h,
                c_h,
                sigma_h,
                k_h,
                a_v,
                b_v,
                c_v,
                sigma_v,
                k_v,
                rho,
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

    h_true = float(params["h_0"])
    v_true = float(params["v_0"])

    h_model = float(params["h_0"])
    v_model = float(params["v_0"])

    a_h = float(params["a_h"])
    b_h = float(params["b_h"])
    c_h = float(params["c_h"])
    sigma_h = float(params["sigma_h"])
    k_h = float(params["k_h"])

    a_v = float(params["a_v"])
    b_v = float(params["b_v"])
    c_v = float(params["c_v"])
    sigma_v = float(params["sigma_v"])
    k_v = float(params["k_v"])

    rho = float(params["rho"])

    rng = np.random.default_rng(int(config.get("seed", 42)))

    times = [0.0]

    h_true_values = [h_true]
    v_true_values = [v_true]

    h_model_values = [h_model]
    v_model_values = [v_model]

    abs_error_h = [0.0]
    abs_error_v = [0.0]

    for step in range(steps):
        xi1 = rng.normal()
        xi2 = rng.normal()

        h_true_next, v_true_next = correlated_aircraft_euler_step(
            h=h_true,
            v=v_true,
            a_h=a_h,
            b_h=b_h,
            c_h=c_h,
            sigma_h=sigma_h,
            k_h=k_h,
            a_v=a_v,
            b_v=b_v,
            c_v=c_v,
            sigma_v=sigma_v,
            k_v=k_v,
            rho=rho,
            dt=dt,
            xi1=xi1,
            xi2=xi2,
        )

        pred_delta_h, pred_delta_v = predict_delta(
            model=model,
            device=device,
            h=h_model,
            v=v_model,
            a_h=a_h,
            b_h=b_h,
            c_h=c_h,
            sigma_h=sigma_h,
            k_h=k_h,
            a_v=a_v,
            b_v=b_v,
            c_v=c_v,
            sigma_v=sigma_v,
            k_v=k_v,
            rho=rho,
            dt=dt,
            xi1=xi1,
            xi2=xi2,
        )

        h_model_next = h_model + pred_delta_h
        v_model_next = v_model + pred_delta_v

        h_true = h_true_next
        v_true = v_true_next

        h_model = h_model_next
        v_model = v_model_next

        current_time = (step + 1) * dt
        times.append(current_time)

        h_true_values.append(h_true)
        v_true_values.append(v_true)

        h_model_values.append(h_model)
        v_model_values.append(v_model)

        abs_error_h.append(abs(h_true - h_model))
        abs_error_v.append(abs(v_true - v_model))

    return {
        "times": np.array(times),
        "h_true": np.array(h_true_values),
        "v_true": np.array(v_true_values),
        "h_model": np.array(h_model_values),
        "v_model": np.array(v_model_values),
        "abs_error_h": np.array(abs_error_h),
        "abs_error_v": np.array(abs_error_v),
    }


def plot_train_log(run_dir):
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
    plt.title("Динаміка навчання моделі для корельованої системи")

    plt.yscale("log")
    plt.yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    plt.gca().yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    path = os.path.join(run_dir, "динаміка_навчання_корельована_система.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_rollout(results, run_dir):
    times = results["times"]

    plt.figure(figsize=(10, 6))
    plt.plot(times, results["h_true"], label="Істинна траєкторія висоти")
    plt.plot(times, results["h_model"], label="Прогноз моделі висоти", linestyle="--")
    plt.xlabel("Час")
    plt.ylabel("Висота H")
    plt.title("Порівняння істинної та прогнозованої траєкторії висоти")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(run_dir, "траєкторія_висоти_корельована_система.png")
    plt.savefig(path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(times, results["v_true"], label="Істинна траєкторія швидкості")
    plt.plot(times, results["v_model"], label="Прогноз моделі швидкості", linestyle="--")
    plt.xlabel("Час")
    plt.ylabel("Швидкість V")
    plt.title("Порівняння істинної та прогнозованої траєкторії швидкості")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(run_dir, "траєкторія_швидкості_корельована_система.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_errors(results, run_dir):
    times = results["times"]

    plt.figure(figsize=(10, 6))
    plt.plot(times, results["abs_error_h"], label="Абсолютна похибка висоти")
    plt.plot(times, results["abs_error_v"], label="Абсолютна похибка швидкості")
    plt.xlabel("Час")
    plt.ylabel("Абсолютна похибка")
    plt.title("Абсолютна похибка прогнозу для корельованої системи")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(run_dir, "абсолютна_похибка_корельована_система.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_phase_trajectory(results, run_dir):
    plt.figure(figsize=(8, 8))
    plt.plot(
        results["h_true"],
        results["v_true"],
        label="Істинна фазова траєкторія",
    )
    plt.plot(
        results["h_model"],
        results["v_model"],
        label="Фазова траєкторія моделі",
        linestyle="--",
    )
    plt.xlabel("Висота H")
    plt.ylabel("Швидкість V")
    plt.title("Фазова траєкторія корельованої системи")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(run_dir, "фазова_траєкторія_корельована_система.png")
    plt.savefig(path, dpi=300)
    plt.close()


def save_rollout_metrics(results, run_dir):
    mean_abs_error_h = float(np.mean(results["abs_error_h"]))
    mean_abs_error_v = float(np.mean(results["abs_error_v"]))

    max_abs_error_h = float(np.max(results["abs_error_h"]))
    max_abs_error_v = float(np.max(results["abs_error_v"]))

    metrics_path = os.path.join(run_dir, "rollout_metrics_correlated_aircraft.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Метрики rollout-оцінювання для корельованої системи\n")
        f.write("=" * 75 + "\n")
        f.write(f"Середня абсолютна похибка висоти H: {mean_abs_error_h:.10f}\n")
        f.write(f"Середня абсолютна похибка швидкості V: {mean_abs_error_v:.10f}\n")
        f.write(f"Максимальна абсолютна похибка висоти H: {max_abs_error_h:.10f}\n")
        f.write(f"Максимальна абсолютна похибка швидкості V: {max_abs_error_v:.10f}\n")

    print(f"Метрики rollout збережено у файл: {metrics_path}")


def monte_carlo_rollout(model, config, device):
    eval_cfg = config["eval"]
    params = eval_cfg["rollout_params"]

    dt = float(eval_cfg["rollout_dt"])
    steps = int(eval_cfg["rollout_steps"])
    n_paths = int(eval_cfg.get("monte_carlo_paths", 200))

    h_0 = float(params["h_0"])
    v_0 = float(params["v_0"])

    a_h = float(params["a_h"])
    b_h = float(params["b_h"])
    c_h = float(params["c_h"])
    sigma_h = float(params["sigma_h"])
    k_h = float(params["k_h"])

    a_v = float(params["a_v"])
    b_v = float(params["b_v"])
    c_v = float(params["c_v"])
    sigma_v = float(params["sigma_v"])
    k_v = float(params["k_v"])

    rho = float(params["rho"])

    rng = np.random.default_rng(int(config.get("seed", 42)))

    times = np.arange(steps + 1) * dt

    true_h = np.zeros((n_paths, steps + 1), dtype=np.float64)
    true_v = np.zeros((n_paths, steps + 1), dtype=np.float64)
    model_h = np.zeros((n_paths, steps + 1), dtype=np.float64)
    model_v = np.zeros((n_paths, steps + 1), dtype=np.float64)

    true_h[:, 0] = h_0
    true_v[:, 0] = v_0
    model_h[:, 0] = h_0
    model_v[:, 0] = v_0

    for path_idx in range(n_paths):
        h_true = h_0
        v_true = v_0
        h_model = h_0
        v_model = v_0

        for step in range(steps):
            xi1 = rng.normal()
            xi2 = rng.normal()

            h_true_next, v_true_next = correlated_aircraft_euler_step(
                h=h_true,
                v=v_true,
                a_h=a_h,
                b_h=b_h,
                c_h=c_h,
                sigma_h=sigma_h,
                k_h=k_h,
                a_v=a_v,
                b_v=b_v,
                c_v=c_v,
                sigma_v=sigma_v,
                k_v=k_v,
                rho=rho,
                dt=dt,
                xi1=xi1,
                xi2=xi2,
            )

            pred_delta_h, pred_delta_v = predict_delta(
                model=model,
                device=device,
                h=h_model,
                v=v_model,
                a_h=a_h,
                b_h=b_h,
                c_h=c_h,
                sigma_h=sigma_h,
                k_h=k_h,
                a_v=a_v,
                b_v=b_v,
                c_v=c_v,
                sigma_v=sigma_v,
                k_v=k_v,
                rho=rho,
                dt=dt,
                xi1=xi1,
                xi2=xi2,
            )

            h_model_next = h_model + pred_delta_h
            v_model_next = v_model + pred_delta_v

            h_true = h_true_next
            v_true = v_true_next
            h_model = h_model_next
            v_model = v_model_next

            true_h[path_idx, step + 1] = h_true
            true_v[path_idx, step + 1] = v_true
            model_h[path_idx, step + 1] = h_model
            model_v[path_idx, step + 1] = v_model

    return {
        "times": times,
        "true_h": true_h,
        "true_v": true_v,
        "model_h": model_h,
        "model_v": model_v,
    }


def plot_mc_mean_std_height(mc_results, run_dir):
    times = mc_results["times"]

    true_mean = np.mean(mc_results["true_h"], axis=0)
    true_std = np.std(mc_results["true_h"], axis=0)

    model_mean = np.mean(mc_results["model_h"], axis=0)
    model_std = np.std(mc_results["model_h"], axis=0)

    plt.figure(figsize=(10, 6))

    plt.plot(times, true_mean, label="Істинне середнє висоти")
    plt.fill_between(
        times,
        true_mean - true_std,
        true_mean + true_std,
        alpha=0.25,
        label="Істинне середнє ± std висоти",
    )

    plt.plot(times, model_mean, linestyle="--", label="Прогнозоване середнє висоти")
    plt.fill_between(
        times,
        model_mean - model_std,
        model_mean + model_std,
        alpha=0.25,
        label="Прогнозоване середнє ± std висоти",
    )

    plt.xlabel("Час")
    plt.ylabel("Висота H")
    plt.title("Monte Carlo: середнє значення та стандартне відхилення для висоти")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(
        run_dir,
        "монте_карло_середнє_std_висота_корельована_система.png",
    )
    plt.savefig(path, dpi=300)
    plt.close()


def plot_mc_mean_std_speed(mc_results, run_dir):
    times = mc_results["times"]

    true_mean = np.mean(mc_results["true_v"], axis=0)
    true_std = np.std(mc_results["true_v"], axis=0)

    model_mean = np.mean(mc_results["model_v"], axis=0)
    model_std = np.std(mc_results["model_v"], axis=0)

    plt.figure(figsize=(10, 6))

    plt.plot(times, true_mean, label="Істинне середнє швидкості")
    plt.fill_between(
        times,
        true_mean - true_std,
        true_mean + true_std,
        alpha=0.25,
        label="Істинне середнє ± std швидкості",
    )

    plt.plot(times, model_mean, linestyle="--", label="Прогнозоване середнє швидкості")
    plt.fill_between(
        times,
        model_mean - model_std,
        model_mean + model_std,
        alpha=0.25,
        label="Прогнозоване середнє ± std швидкості",
    )

    plt.xlabel("Час")
    plt.ylabel("Швидкість V")
    plt.title("Monte Carlo: середнє значення та стандартне відхилення для швидкості")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(
        run_dir,
        "монте_карло_середнє_std_швидкість_корельована_система.png",
    )
    plt.savefig(path, dpi=300)
    plt.close()

def plot_mc_phase_ensemble(mc_results, run_dir, n_paths_to_plot=12):
    true_h = mc_results["true_h"]
    true_v = mc_results["true_v"]
    model_h = mc_results["model_h"]
    model_v = mc_results["model_v"]

    n_available = true_h.shape[0]
    n_plot = min(n_paths_to_plot, n_available)

    plt.figure(figsize=(9, 9))

    for i in range(n_plot):
        plt.plot(
            true_h[i],
            true_v[i],
            color="tab:blue",
            alpha=0.35,
            linewidth=1.2,
            label="Істинні фазові траєкторії" if i == 0 else None,
        )

        plt.plot(
            model_h[i],
            model_v[i],
            color="tab:orange",
            linestyle="--",
            alpha=0.35,
            linewidth=1.2,
            label="Прогнозовані фазові траєкторії" if i == 0 else None,
        )

    plt.xlabel("Висота H")
    plt.ylabel("Швидкість V")
    plt.title("Monte Carlo: ансамбль фазових траєкторій")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(
        run_dir,
        "монте_карло_фазові_траєкторії_корельована_система.png",
    )
    plt.savefig(path, dpi=300)
    plt.close()

def plot_mc_terminal_phase_space(mc_results, run_dir):
    true_h_terminal = mc_results["true_h"][:, -1]
    true_v_terminal = mc_results["true_v"][:, -1]

    model_h_terminal = mc_results["model_h"][:, -1]
    model_v_terminal = mc_results["model_v"][:, -1]

    plt.figure(figsize=(8, 8))

    plt.scatter(
        true_h_terminal,
        true_v_terminal,
        alpha=0.6,
        s=18,
        label="Істинні кінцеві стани",
    )

    plt.scatter(
        model_h_terminal,
        model_v_terminal,
        alpha=0.6,
        s=18,
        label="Прогнозовані кінцеві стани",
        marker="x",
    )

    plt.xlabel("H(T)")
    plt.ylabel("V(T)")
    plt.title("Monte Carlo: розподіл кінцевих станів у фазовому просторі")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(
        run_dir,
        "монте_карло_кінцеві_стани_фазовий_простір_корельована_система.png",
    )
    plt.savefig(path, dpi=300)
    plt.close()


def save_mc_metrics(mc_results, run_dir):
    true_mean_h_final = float(np.mean(mc_results["true_h"][:, -1]))
    true_std_h_final = float(np.std(mc_results["true_h"][:, -1]))

    model_mean_h_final = float(np.mean(mc_results["model_h"][:, -1]))
    model_std_h_final = float(np.std(mc_results["model_h"][:, -1]))

    true_mean_v_final = float(np.mean(mc_results["true_v"][:, -1]))
    true_std_v_final = float(np.std(mc_results["true_v"][:, -1]))

    model_mean_v_final = float(np.mean(mc_results["model_v"][:, -1]))
    model_std_v_final = float(np.std(mc_results["model_v"][:, -1]))

    metrics_path = os.path.join(run_dir, "mc_metrics_correlated_aircraft.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Monte Carlo-метрики для корельованої системи\n")
        f.write("=" * 75 + "\n")
        f.write(f"Істинне середнє H(T): {true_mean_h_final:.10f}\n")
        f.write(f"Істинне std H(T): {true_std_h_final:.10f}\n")
        f.write(f"Прогнозоване середнє H(T): {model_mean_h_final:.10f}\n")
        f.write(f"Прогнозоване std H(T): {model_std_h_final:.10f}\n")
        f.write("\n")
        f.write(f"Істинне середнє V(T): {true_mean_v_final:.10f}\n")
        f.write(f"Істинне std V(T): {true_std_v_final:.10f}\n")
        f.write(f"Прогнозоване середнє V(T): {model_mean_v_final:.10f}\n")
        f.write(f"Прогнозоване std V(T): {model_std_v_final:.10f}\n")

    print(f"Monte Carlo-метрики збережено у файл: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Оцінювання моделі для корельованої стохастичної системи"
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Папка запуску. Якщо не вказано, буде використано останній correlated_aircraft_run.",
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

    print("Починаємо оцінювання моделі для корельованої системи")
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

    plot_mc_mean_std_height(mc_results, run_dir)
    plot_mc_mean_std_speed(mc_results, run_dir)
    plot_mc_terminal_phase_space(mc_results, run_dir)
    plot_mc_phase_ensemble(mc_results, run_dir)
    save_mc_metrics(mc_results, run_dir)

    print("Оцінювання завершено.")
    print("Побудовано графіки:")
    print("- динаміка_навчання_корельована_система.png")
    print("- траєкторія_висоти_корельована_система.png")
    print("- траєкторія_швидкості_корельована_система.png")
    print("- абсолютна_похибка_корельована_система.png")
    print("- фазова_траєкторія_корельована_система.png")
    print("- монте_карло_середнє_std_висота_корельована_система.png")
    print("- монте_карло_середнє_std_швидкість_корельована_система.png")
    print("- монте_карло_кінцеві_стани_фазовий_простір_корельована_система.png")
    print("- монте_карло_фазові_траєкторії_корельована_система.png")    

if __name__ == "__main__":
    main()