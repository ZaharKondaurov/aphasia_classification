import torch.nn as nn
from transformers import HubertForSequenceClassification, HubertModel


class HuBERTClassifier(nn.Module):

    def __init__(self, num_labels: int = 2, unfreeze: float = 0.5):
        super(HuBERTClassifier, self).__init__()

        self.hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960",
            # num_labels=num_labels,
        )

        self.num_parameters = len(list(self.hubert.parameters()))

        for ind, param in enumerate(self.hubert.parameters()):

            if ind + 4 < int(self.num_parameters * unfreeze):
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(768, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, audio):
        x = self.hubert(audio).last_hidden_state

        x = x.permute(0, 2, 1)
        x_emb = self.avg_pool(x).squeeze(-1)

        x = self.fc(x_emb)
        return self.softmax(x), x_emb
