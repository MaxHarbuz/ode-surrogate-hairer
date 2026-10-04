import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader


def ou_euler_step(x, theta, mu, sigma, dt, xi):
    """
    Один крок методу Ейлера-Маруями для процесу Орнштейна-Уленбека:

        dX = theta * (mu - X) dt + sigma dW

    Дискретна форма:

        X(t + dt) = X(t) + theta * (mu - X(t)) * dt + sigma * sqrt(dt) * xi

    Тут xi - стандартна нормальна випадкова величина.
    """

    x_next = x + theta * (mu - x) * dt + sigma * np.sqrt(dt) * xi
    return x_next


def generate_ou_arrays(config):
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    dcfg = config["data"]

    n = (
        int(dcfg["train_samples"])
        + int(dcfg["val_samples"])
        + int(dcfg["test_samples"])
    )

    x = rng.uniform(
        float(dcfg["x_min"]),
        float(dcfg["x_max"]),
        size=n,
    )

    theta = rng.uniform(
        float(dcfg["theta_min"]),
        float(dcfg["theta_max"]),
        size=n,
    )

    mu = rng.uniform(
        float(dcfg["mu_min"]),
        float(dcfg["mu_max"]),
        size=n,
    )

    sigma = rng.uniform(
        float(dcfg["sigma_min"]),
        float(dcfg["sigma_max"]),
        size=n,
    )

    dt = rng.uniform(
        float(dcfg["dt_min"]),
        float(dcfg["dt_max"]),
        size=n,
    )

    xi = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )

    x_next = ou_euler_step(
        x=x,
        theta=theta,
        mu=mu,
        sigma=sigma,
        dt=dt,
        xi=xi,
    )

    delta_x = x_next - x

    inputs = np.stack(
        [
            x,
            theta,
            mu,
            sigma,
            dt,
            xi,
        ],
        axis=1,
    ).astype(np.float32)

    targets = delta_x.reshape(-1, 1).astype(np.float32)

    return inputs, targets


def save_ou_npz(path: str, inputs: np.ndarray, targets: np.ndarray, config: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    dcfg = config["data"]

    np.savez_compressed(
        path,
        inputs=inputs,
        targets=targets,
        seed=int(config.get("seed", 42)),
        x_min=float(dcfg["x_min"]),
        x_max=float(dcfg["x_max"]),
        theta_min=float(dcfg["theta_min"]),
        theta_max=float(dcfg["theta_max"]),
        mu_min=float(dcfg["mu_min"]),
        mu_max=float(dcfg["mu_max"]),
        sigma_min=float(dcfg["sigma_min"]),
        sigma_max=float(dcfg["sigma_max"]),
        dt_min=float(dcfg["dt_min"]),
        dt_max=float(dcfg["dt_max"]),
    )


def load_ou_npz(path: str):
    data = np.load(path)
    return data["inputs"], data["targets"]


class OUDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_ou_splits(dataset: Dataset, config: dict):
    train_samples = int(config["data"]["train_samples"])
    val_samples = int(config["data"]["val_samples"])
    test_samples = int(config["data"]["test_samples"])

    expected = train_samples + val_samples + test_samples

    if len(dataset) != expected:
        raise ValueError(
            f"Розмір датасету не відповідає конфігурації: "
            f"len(dataset)={len(dataset)}, expected={expected}"
        )

    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_samples, val_samples, test_samples],
        generator=generator,
    )

    return train_ds, val_ds, test_ds


def make_ou_loaders(train_ds, val_ds, test_ds, config: dict):
    batch_size = int(config["train"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader