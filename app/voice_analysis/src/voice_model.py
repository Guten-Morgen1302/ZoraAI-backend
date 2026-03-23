import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            nn.BatchNorm2d(out_c), 
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride), 
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class ResNetBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone extracts spectral textures
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            ResidualBlock(32, 64, stride=2), # Freq dim -> 20
            ResidualBlock(64, 128, stride=2) # Freq dim -> 10
        )
        
        # Bi-LSTM extracts temporal speech flow
        # input_size = channels(128) * freq_bins(10)
        self.lstm = nn.LSTM(
            input_size=128 * 10, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Classification head (256 because of Bidirectional LSTM)
        self.fc = nn.Sequential(
            nn.Linear(256, 64), 
            nn.ReLU(), 
            nn.Dropout(0.4), 
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        b, c, f, t = x.size()
        
        # Reshape for LSTM: [Batch, Time, Features]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        
        x, _ = self.lstm(x)
        
        # Take the output of the last time step
        return self.fc(x[:, -1, :])