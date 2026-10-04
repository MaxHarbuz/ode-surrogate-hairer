import torch.nn as nn


class OUSurrogate(nn.Module):
    """
    Нейромережева сурогатна модель для одновимірного
    процесу Орнштейна-Уленбека.

    Вхід моделі:
        [x, theta, mu, sigma, dt, xi]

    Вихід моделі:
        [delta_x]
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 1,
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