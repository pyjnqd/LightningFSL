import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



class MN_head(nn.Module):

    def __init__(
        self) -> None:
        super().__init__()

        self.softmax = nn.Softmax()

    def forward(self, features_test: Tensor, features_train: Tensor,
                way: int, shot: int) -> Tensor:
        r"""Take batches of few-shot training examples and testing examples as input,
            output the logits of each testing examples.

        Args:
            features_test: Testing examples. size: [batch_size, num_query, c, h, w]
            features_train: Training examples which has labels like:[abcdabcdabcd].
                            size: [batch_size, way*shot, c, h, w]
            way: The number of classes of each few-shot classification task.
            shot: The number of training images per class in each few-shot classification
                  task.
        Output:
            classification_scores: The calculated logits of testing examples.
                                   size: [batch_size, num_query, way]
        """
        if features_train.dim() == 5:
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
        assert features_train.dim() == 3

        # batch_size = features_train.size(0)

        features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        # #prototypes: [batch_size, way, c]
        # prototypes = torch.mean(features_train.reshape(batch_size, shot, way, -1),dim=1)
        # prototypes = F.normalize(prototypes, p=2, dim=2, eps=1e-12)

        if features_test.dim() == 5:
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)

        assert features_test.dim() == 3
        features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)

        scores = features_test.bmm(features_train.transpose(1,2)) # The original paper use cosine simlarity, but here we scale it by 100 to strengthen highest probability after softmax

        classification = torch.mean(scores.reshape(features_test.size(0), features_test.size(1), shot, way), dim=2)
        return classification


def create_model():
    return MN_head()
