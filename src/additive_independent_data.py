import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader


def additive_independent_euler_step(
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
    """
    Один крок методу Ейлера-Маруями для двовимірної
    адитивної незалежної стохастичної системи:

        dX1 = (a1 * X1 + b1) dt + sigma1 dW1
        dX2 = (a2 * X2 + b2) dt + sigma2 dW2

    Дискретна форма:

        X1(t + dt) = X1(t) + (a1 * X1(t) + b1) * dt + sigma1 * sqrt(dt) * xi1
        X2(t + dt) = X2(t) + (a2 * X2(t) + b2) * dt + sigma2 * sqrt(dt) * xi2

    Тут xi1 та xi2 — незалежні стандартні нормальні випадкові величини.
    """

    x1_next = x1 + (a1 * x1 + b1) * dt + sigma1 * np.sqrt(dt) * xi1
    x2_next = x2 + (a2 * x2 + b2) * dt + sigma2 * np.sqrt(dt) * xi2

    return x1_next, x2_next


def generate_additive_independent_arrays(config):
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    dcfg = config["data"]

    n = (
        int(dcfg["train_samples"])
        + int(dcfg["val_samples"])
        + int(dcfg["test_samples"])
    )

    x1 = rng.uniform(
        float(dcfg["x1_min"]),
        float(dcfg["x1_max"]),
        size=n,
    )

    x2 = rng.uniform(
        float(dcfg["x2_min"]),
        float(dcfg["x2_max"]),
        size=n,
    )

    a1 = rng.uniform(
        float(dcfg["a1_min"]),
        float(dcfg["a1_max"]),
        size=n,
    )

    b1 = rng.uniform(
        float(dcfg["b1_min"]),
        float(dcfg["b1_max"]),
        size=n,
    )

    sigma1 = rng.uniform(
        float(dcfg["sigma1_min"]),
        float(dcfg["sigma1_max"]),
        size=n,
    )

    a2 = rng.uniform(
        float(dcfg["a2_min"]),
        float(dcfg["a2_max"]),
        size=n,
    )

    b2 = rng.uniform(
        float(dcfg["b2_min"]),
        float(dcfg["b2_max"]),
        size=n,
    )

    sigma2 = rng.uniform(
        float(dcfg["sigma2_min"]),
        float(dcfg["sigma2_max"]),
        size=n,
    )

    dt = rng.uniform(
        float(dcfg["dt_min"]),
        float(dcfg["dt_max"]),
        size=n,
    )

    xi1 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )

    xi2 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )

    x1_next, x2_next = additive_independent_euler_step(
        x1=x1,
        x2=x2,
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

    delta_x1 = x1_next - x1
    delta_x2 = x2_next - x2

    inputs = np.stack(
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
        ],
        axis=1,
    ).astype(np.float32)

    targets = np.stack(
        [
            delta_x1,
            delta_x2,
        ],
        axis=1,
    ).astype(np.float32)

    return inputs, targets


def save_additive_independent_npz(
    path: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: dict,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    dcfg = config["data"]

    np.savez_compressed(
        path,
        inputs=inputs,
        targets=targets,
        seed=int(config.get("seed", 42)),
        x1_min=float(dcfg["x1_min"]),
        x1_max=float(dcfg["x1_max"]),
        x2_min=float(dcfg["x2_min"]),
        x2_max=float(dcfg["x2_max"]),
        a1_min=float(dcfg["a1_min"]),
        a1_max=float(dcfg["a1_max"]),
        b1_min=float(dcfg["b1_min"]),
        b1_max=float(dcfg["b1_max"]),
        sigma1_min=float(dcfg["sigma1_min"]),
        sigma1_max=float(dcfg["sigma1_max"]),
        a2_min=float(dcfg["a2_min"]),
        a2_max=float(dcfg["a2_max"]),
        b2_min=float(dcfg["b2_min"]),
        b2_max=float(dcfg["b2_max"]),
        sigma2_min=float(dcfg["sigma2_min"]),
        sigma2_max=float(dcfg["sigma2_max"]),
        dt_min=float(dcfg["dt_min"]),
        dt_max=float(dcfg["dt_max"]),
    )


def load_additive_independent_npz(path: str):
    data = np.load(path)
    return data["inputs"], data["targets"]


class AdditiveIndependentDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_additive_independent_splits(dataset: Dataset, config: dict):
    train_samples = int(config["data"]["train_samples"])
    val_samples = int(config["data"]["val_samples"])
    test_samples = int(config["data"]["test_samples"])

    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))

    return random_split(
        dataset,
        [train_samples, val_samples, test_samples],
        generator=generator,
    )


def make_additive_independent_loaders(train_ds, val_ds, test_ds, config):
    batch_size = int(config["train"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader