import os
import sys
import subprocess


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_command(command):
    print("=" * 80)
    print("Виконується команда:")
    print(" ".join(command))
    print("=" * 80)

    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"Команда завершилася з помилкою: {' '.join(command)}"
        )


def main():
    config_path = "configs/correlated_aircraft.yaml"

    run_command(
        [
            sys.executable,
            "scripts/make_correlated_aircraft_dataset.py",
            "--config",
            config_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/train_correlated_aircraft.py",
            "--config",
            config_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/eval_correlated_aircraft.py",
        ]
    )

    print("=" * 80)
    print("Експеримент для корельованої системи літака повністю завершено.")
    print("=" * 80)


if __name__ == "__main__":
    main()