from torch import nn
from torch.nn import functional as F
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor


class Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_test: Tensor, features_train: Tensor,
                way: int, shot: int):
        assert features_train.size(0) == features_test.size(0) == 1

        features_train = torch.squeeze(features_train, 0)
        features_test = torch.squeeze(features_test, 0)

        if features_train.dim() == 4:
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
        # note
        features_train = F.normalize(features_train, p=2, dim=1, eps=1e-12)
        features_test = F.normalize(features_test, p=2, dim=1, eps=1e-12)
        assert features_train.dim() == features_test.dim() == 2


        X_sup = features_train.cpu().detach().numpy()
        X_query = features_test.cpu().detach().numpy()
        label = torch.arange(way, dtype=torch.int8).repeat(shot).numpy()

        classifier = LogisticRegression(random_state=0, max_iter=1000).fit(X=X_sup, y=label)
        classification_scores = torch.from_numpy(classifier.predict_proba(X_query)).to(features_test.device)
        return classification_scores


def create_model():
    return Logistic_Regression()
