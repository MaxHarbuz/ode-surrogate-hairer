import torch
import torch.nn as nn


class CorrelatedAircraftSurrogate(nn.Module):
    """
    Нейромережева сурогатна модель для корельованої двовимірної
    стохастичної системи літака зі зв'язаними drift-компонентами
    та мультиплікативним шумом.

    Вхід моделі:
        [
            h, v,
            a_h, b_h, c_h, sigma_h, k_h,
            a_v, b_v, c_v, sigma_v, k_v,
            rho, dt, xi1, xi2
        ]

    Вихід моделі:
        [delta_h, delta_v]
    """

    def __init__(
        self,
        input_dim: int = 16,
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