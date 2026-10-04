import torch
import torch.nn as nn


class AdditiveIndependentSurrogate(nn.Module):
    """
    Нейромережева сурогатна модель для двовимірної
    адитивної незалежної стохастичної системи.

    Вхід моделі:
        [x1, x2, a1, b1, sigma1, a2, b2, sigma2, dt, xi1, xi2]

    Вихід моделі:
        [delta_x1, delta_x2]
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        output_dim: int = 2,
        n_layers: int = 3,
        activation: str = "gelu",
    ):
        super().__init__()

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "tanh":
            act_layer = nn.Tanh
        elif activation == "gelu":
            act_layer = nn.GELU
        else:
            raise ValueError(f"Непідтримувана функція активації: {activation}")

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_layer())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_layer())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)