import os
import sys
import subprocess


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_command(command):
    print("=" * 90)
    print("Виконується команда:")
    print(" ".join(command))
    print("=" * 90)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    return result


def main():
    config_path = "configs/ou.yaml"

    python_executable = sys.executable

    print("Починаємо повний експеримент для процесу Орнштейна-Уленбека...")

    run_command(
        [
            python_executable,
            "scripts/make_ou_dataset.py",
            "--config",
            config_path,
        ]
    )

    run_command(
        [
            python_executable,
            "scripts/train_ou.py",
            "--config",
            config_path,
        ]
    )

    run_command(
        [
            python_executable,
            "scripts/eval_ou.py",
        ]
    )

    print("=" * 90)
    print("Експеримент для процесу Орнштейна-Уленбека повністю завершено.")
    print("=" * 90)


if __name__ == "__main__":
    main()