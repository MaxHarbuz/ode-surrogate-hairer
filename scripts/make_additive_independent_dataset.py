import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import argparse
import yaml

from src.additive_independent_data import (
    generate_additive_independent_arrays,
    save_additive_independent_npz,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Генерація датасету для адитивної незалежної стохастичної системи"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/additive_independent.yaml",
        help="Шлях до конфігураційного файлу",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    dataset_path = config["paths"]["dataset_path"]

    print("Починаємо генерацію датасету для адитивної незалежної системи...")
    print(f"Файл конфігурації: {args.config}")
    print(f"Шлях збереження датасету: {dataset_path}")

    inputs, targets = generate_additive_independent_arrays(config)

    save_additive_independent_npz(
        path=dataset_path,
        inputs=inputs,
        targets=targets,
        config=config,
    )

    print("Датасет успішно згенеровано.")
    print(f"Кількість прикладів: {inputs.shape[0]}")
    print(f"Розмірність входу: {inputs.shape[1]}")
    print(f"Розмірність виходу: {targets.shape[1]}")
    print(f"Датасет збережено у файл: {dataset_path}")


if __name__ == "__main__":
    main()