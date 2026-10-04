import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader


def correlated_aircraft_euler_step(
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
    """
    Один крок методу Ейлера-Маруями для корельованої двовимірної
    стохастичної системи зі зв'язаними drift-компонентами
    та мультиплікативним шумом.

    Система:

        dH = (a_h * H + b_h * V + c_h) dt
             + sigma_h * (1 + k_h * |H|) dW1

        dV = (a_v * V + b_v * H + c_v) dt
             + sigma_v * (1 + k_v * |V|) dW2

        corr(dW1, dW2) = rho

    Корельовані шуми будуються так:

        z1 = xi1
        z2 = rho * xi1 + sqrt(1 - rho^2) * xi2

    де xi1, xi2 — незалежні стандартні нормальні випадкові величини.
    """

    rho = np.clip(rho, -0.999, 0.999)

    z1 = xi1
    z2 = rho * xi1 + np.sqrt(1.0 - rho ** 2) * xi2

    drift_h = a_h * h + b_h * v + c_h
    drift_v = a_v * v + b_v * h + c_v

    diffusion_h = sigma_h * (1.0 + k_h * np.abs(h))
    diffusion_v = sigma_v * (1.0 + k_v * np.abs(v))

    h_next = h + drift_h * dt + diffusion_h * np.sqrt(dt) * z1
    v_next = v + drift_v * dt + diffusion_v * np.sqrt(dt) * z2

    return h_next, v_next


def generate_correlated_aircraft_arrays(config):
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    dcfg = config["data"]

    n = (
        int(dcfg["train_samples"])
        + int(dcfg["val_samples"])
        + int(dcfg["test_samples"])
    )

    h = rng.uniform(float(dcfg["h_min"]), float(dcfg["h_max"]), size=n)
    v = rng.uniform(float(dcfg["v_min"]), float(dcfg["v_max"]), size=n)

    a_h = rng.uniform(float(dcfg["a_h_min"]), float(dcfg["a_h_max"]), size=n)
    b_h = rng.uniform(float(dcfg["b_h_min"]), float(dcfg["b_h_max"]), size=n)
    c_h = rng.uniform(float(dcfg["c_h_min"]), float(dcfg["c_h_max"]), size=n)
    sigma_h = rng.uniform(float(dcfg["sigma_h_min"]), float(dcfg["sigma_h_max"]), size=n)
    k_h = rng.uniform(float(dcfg["k_h_min"]), float(dcfg["k_h_max"]), size=n)

    a_v = rng.uniform(float(dcfg["a_v_min"]), float(dcfg["a_v_max"]), size=n)
    b_v = rng.uniform(float(dcfg["b_v_min"]), float(dcfg["b_v_max"]), size=n)
    c_v = rng.uniform(float(dcfg["c_v_min"]), float(dcfg["c_v_max"]), size=n)
    sigma_v = rng.uniform(float(dcfg["sigma_v_min"]), float(dcfg["sigma_v_max"]), size=n)
    k_v = rng.uniform(float(dcfg["k_v_min"]), float(dcfg["k_v_max"]), size=n)

    rho = rng.uniform(float(dcfg["rho_min"]), float(dcfg["rho_max"]), size=n)

    dt = rng.uniform(float(dcfg["dt_min"]), float(dcfg["dt_max"]), size=n)

    xi1 = rng.normal(loc=0.0, scale=1.0, size=n)
    xi2 = rng.normal(loc=0.0, scale=1.0, size=n)

    h_next, v_next = correlated_aircraft_euler_step(
        h=h,
        v=v,
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

    delta_h = h_next - h
    delta_v = v_next - v

    inputs = np.stack(
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
        ],
        axis=1,
    ).astype(np.float32)

    targets = np.stack(
        [
            delta_h,
            delta_v,
        ],
        axis=1,
    ).astype(np.float32)

    return inputs, targets


def save_correlated_aircraft_npz(
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

        h_min=float(dcfg["h_min"]),
        h_max=float(dcfg["h_max"]),
        v_min=float(dcfg["v_min"]),
        v_max=float(dcfg["v_max"]),

        a_h_min=float(dcfg["a_h_min"]),
        a_h_max=float(dcfg["a_h_max"]),
        b_h_min=float(dcfg["b_h_min"]),
        b_h_max=float(dcfg["b_h_max"]),
        c_h_min=float(dcfg["c_h_min"]),
        c_h_max=float(dcfg["c_h_max"]),
        sigma_h_min=float(dcfg["sigma_h_min"]),
        sigma_h_max=float(dcfg["sigma_h_max"]),
        k_h_min=float(dcfg["k_h_min"]),
        k_h_max=float(dcfg["k_h_max"]),

        a_v_min=float(dcfg["a_v_min"]),
        a_v_max=float(dcfg["a_v_max"]),
        b_v_min=float(dcfg["b_v_min"]),
        b_v_max=float(dcfg["b_v_max"]),
        c_v_min=float(dcfg["c_v_min"]),
        c_v_max=float(dcfg["c_v_max"]),
        sigma_v_min=float(dcfg["sigma_v_min"]),
        sigma_v_max=float(dcfg["sigma_v_max"]),
        k_v_min=float(dcfg["k_v_min"]),
        k_v_max=float(dcfg["k_v_max"]),

        rho_min=float(dcfg["rho_min"]),
        rho_max=float(dcfg["rho_max"]),
        dt_min=float(dcfg["dt_min"]),
        dt_max=float(dcfg["dt_max"]),
    )


def load_correlated_aircraft_npz(path: str):
    data = np.load(path)
    return data["inputs"], data["targets"]


class CorrelatedAircraftDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_correlated_aircraft_splits(dataset: Dataset, config: dict):
    train_samples = int(config["data"]["train_samples"])
    val_samples = int(config["data"]["val_samples"])
    test_samples = int(config["data"]["test_samples"])

    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))

    return random_split(
        dataset,
        [train_samples, val_samples, test_samples],
        generator=generator,
    )


def make_correlated_aircraft_loaders(train_ds, val_ds, test_ds, config):
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