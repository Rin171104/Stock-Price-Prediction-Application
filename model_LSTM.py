import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=50,
        num_layers=2,
        dropout=0.2,
        output_size=1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected block
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        out, _ = self.lstm(x)

        # Lấy output của timestep cuối
        out = out[:, -1, :]

        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = LSTMModel()

    # batch_size=8, seq_len=60, input_size=1
    input_data = torch.rand(8, 60, 1)

    if torch.cuda.is_available():
        model.cuda()
        input_data = input_data.cuda()

    while True:
        output = model(input_data)
        print("Output shape:", output.shape)
        break
