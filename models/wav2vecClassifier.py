import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification

class Wav2vecClassifier(nn.Module):

    def __init__(self, num_labels: int = 2, unfreeze: float = 0.5):
        super(Wav2vecClassifier, self).__init__()

        self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_labels,
        )

        self.num_parameters = len(list(self.wav2vec.parameters()))

        for ind, param in enumerate(self.wav2vec.parameters()):

            if ind + 4 < int(self.num_parameters * unfreeze):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, audio):
        return self.wav2vec(audio)
