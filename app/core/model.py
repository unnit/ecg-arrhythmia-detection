import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        # CNN blocks extract local morphological features from the waveform
        cnn_layers = []
        in_ch = 1
        for out_ch in cfg["cnn_filters"]:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cfg["kernel_size"], padding=cfg["kernel_size"] // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(cfg["dropout"] * 0.5),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # BiLSTM captures temporal dependencies across the beat sequence
        self.bilstm = nn.LSTM(
            input_size=cfg["cnn_filters"][-1],
            hidden_size=cfg["lstm_hidden"],
            num_layers=cfg["lstm_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=cfg["dropout"] if cfg["lstm_layers"] > 1 else 0.0,
        )

        # attention weights which timesteps matter most
        self.attention = nn.Linear(cfg["lstm_hidden"] * 2, 1)

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(cfg["lstm_hidden"] * 2, 64),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(64, cfg["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn).sum(dim=1)
        return self.classifier(context)
