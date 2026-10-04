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
    config_path = "configs/additive_independent.yaml"

    run_command(
        [
            sys.executable,
            "scripts/make_additive_independent_dataset.py",
            "--config",
            config_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/train_additive_independent.py",
            "--config",
            config_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/eval_additive_independent.py",
        ]
    )

    print("=" * 80)
    print("Експеримент для адитивної незалежної системи повністю завершено.")
    print("=" * 80)


if __name__ == "__main__":
    main()