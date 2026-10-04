import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import yaml

from src.ou_data import (
    generate_ou_arrays,
    save_ou_npz,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Генерація датасету для процесу Орнштейна-Уленбека"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ou.yaml",
        help="Шлях до конфігураційного файлу",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    dataset_path = config["paths"]["dataset_path"]

    print("Генеруємо датасет для процесу Орнштейна-Уленбека...")
    print(f"Файл конфігурації: {args.config}")
    print(f"Шлях збереження датасету: {dataset_path}")

    inputs, targets = generate_ou_arrays(config)

    save_ou_npz(
        path=dataset_path,
        inputs=inputs,
        targets=targets,
        config=config,
    )

    print("Датасет успішно створено.")
    print(f"inputs.shape = {inputs.shape}")
    print(f"targets.shape = {targets.shape}")


if __name__ == "__main__":
    main()