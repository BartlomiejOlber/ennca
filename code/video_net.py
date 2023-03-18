import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VideoNet(nn.Module):
    def __init__(self, cnn=None, lstm=None, attn=None):
        super().__init__()
        if cnn:
            self.cnn = cnn
        else:
            self.cnn = models.resnet18()
            self.cnn.fc = nn.Sequential()

        self.lstm = lstm if lstm else nn.LSTM(input_size=512, hidden_size=128)
        self.attn = attn if attn else nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.use_lstm = ~bool(attn)
        if self.use_lstm:
            self.cls = nn.Linear(128, 8)
        else:
            self.cls = nn.Linear(512, 8)

    def forward(self, sequence):
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence)
        assert sequence.dim() == 4
        features = self.cnn(sequence)
        if self.use_lstm:
            temporal_features, _ = self.lstm(features)  # assume unbatched input where frames.shape[0] is seq_length
            cls = F.log_softmax(self.cls(temporal_features[-1]), dim=1)
        else:
            cls_token = nn.Parameter(torch.zeros(1, features.shape[-1]))
            features = torch.cat((cls_token, features), dim=0)
            temporal_features, _ = self.attn(features, features, features)
            cls = F.log_softmax(self.cls(temporal_features[-1]), dim=0)
        return cls
