# from .base_module import BaseFewShotModule
import torch.nn as nn
from typing import Tuple, List, Optional, Union, Dict
import torch.nn.functional as F
# from architectures import get_classifier, get_backbone
import torch
from torchmetrics import Accuracy, AverageMeter

"""Reference: Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
class SupCluLoss(nn.Module):
    def __init__(self, temperature=0.3, contrast_mode='all',
                 base_temperature=0.07):
        super(SupCluLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = 1
        contrast_feature = features
        if self.contrast_mode == 'one':
            assert False
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# class Jigsaw(BaseFewShotModule):
#
#
#     def __init__(
#             self,
#             mlp_dim: int = 128,
#             task_classifier_name: str = "proto_head",
#             task_classifier_params: Dict = {"learn_scale": False},
#             backbone_name: str = "resnet12",
#             way: int = 5,
#             val_shot: int = 5,
#             test_shot: int = 5,
#             num_query: int = 15,
#             val_batch_size_per_gpu: int = 2,
#             test_batch_size_per_gpu: int = 2,
#             lr: float = 0.1,
#             weight_decay: float = 5e-4,
#             decay_scheduler: Optional[str] = "cosine",
#             optim_type: str = "sgd",
#             decay_epochs: Union[List, Tuple, None] = None,
#             decay_power: Optional[float] = None,
#             backbone_kwargs: Dict = {},
#             **kwargs
#     ) -> None:
#         """
#         Args:
#             mlp_dim: The dimension of the MLP head after the backbone network.
#             task_classifier_name: The name of the classifier for downstream (val, test)
#                                   few-shot tasks. It should match the name of file that
#                                   contains the classifier class.
#             task_classifier_params: The initial parameters of the classifier for
#                                     downstream (val, test) few-shot tasks.
#             backbone_name: The name of the feature extractor,
#                         which should match the correspond
#                         file name in architectures.feature_extractor
#             way: The number of classes within one task.
#             val_shot: The number of samples within each few-shot
#                     support class during validation.
#             test_shot: The number of samples within each few-shot
#                     support class during testing.
#             num_query: The number of samples within each few-shot
#                     query class.
#             val_batch_size_per_gpu: The batch size of validation per GPU.
#             test_batch_size_per_gpu: The batch size of testing per GPU.
#             lr: The initial learning rate.
#             weight_decay: The weight decay parameter.
#             decay_scheduler: The scheduler of optimizer.
#                             "cosine" or "specified_epochs".
#             optim_type: The optimizer type.
#                         "sgd" or "adam"
#             decay_epochs: The list of decay epochs of decay_scheduler "specified_epochs".
#             decay_power: The decay power of decay_scheduler "specified_epochs"
#                         at eachspeicified epoch.
#                         i.e., adjusted_lr = lr * decay_power
#             backbone_kwargs: The parameters for creating backbone network.
#         """
#         super().__init__(
#             backbone_name=backbone_name, way=way, val_shot=val_shot,
#             test_shot=test_shot, num_query=num_query,
#             val_batch_size_per_gpu=val_batch_size_per_gpu, test_batch_size_per_gpu=test_batch_size_per_gpu,
#             lr=lr, weight_decay=weight_decay, decay_scheduler=decay_scheduler, optim_type=optim_type,
#             decay_epochs=decay_epochs, decay_power=decay_power, backbone_kwargs=backbone_kwargs
#         )
#         self.save_hyperparameters()
#         self.criterion_clu = SupCluLoss(temperature=self.hparams.temparature)
#         self.criterion_loc = nn.CrossEntropyLoss()
#         self.fc_clu = nn.Sequential(
#             nn.Linear(self.backbone.outdim, self.backbone.outdim), nn.ReLU(), nn.Linear(self.backbone.outdim, mlp_dim)
#         )
#         self.fc_loc = nn.Linear(self.backbone.outdim, 4)
#         self.val_test_classifier = get_classifier(task_classifier_name, **task_classifier_params)
#         self.ada_avg_pool2d = nn.AdaptiveAvgPool2d((2, 2))
#         self.ada_max_pool2d = nn.AdaptiveMaxPool2d((2, 2))
#
#     @torch.no_grad()
#     def _batch_gather(self, images):
#         """
#         batch_gather in single gpu
#         """
#         images_gather = images
#         n, c, h, w = images_gather[0].shape
#         permute = torch.randperm(n * 4).cuda()
#         images_gather = torch.cat(images_gather, dim=0)
#         images_gather = images_gather[permute, :, :, :]
#         col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
#         col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:]], dim=3)
#         images_gather = torch.cat([col1, col2], dim=2)
#
#         return images_gather, permute, n
#
#     @torch.no_grad()
#     def _batch_gather_ddp(self, images):
#         """
#         gather images from different gpus and shuffle between them
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         images_gather = []
#         for i in range(4):
#             batch_size_this = images[i].shape[0] # 一个GPU上的batchsize
#             images_gather.append(concat_all_gather(images[i]))
#             batch_size_all = images_gather[i].shape[0]
#         num_gpus = batch_size_all // batch_size_this
#
#         n, c, h, w = images_gather[0].shape
#         permute = torch.randperm(n * 4).cuda()
#         torch.distributed.broadcast(permute, src=0)
#         images_gather = torch.cat(images_gather, dim=0)
#         images_gather = images_gather[permute, :, :, :]
#         col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
#         col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:]], dim=3)
#         images_gather = torch.cat([col1, col2], dim=2)
#
#         bs = images_gather.shape[0] // num_gpus
#         gpu_idx = torch.distributed.get_rank()
#
#         return images_gather[bs * gpu_idx:bs * (gpu_idx + 1)], permute, n
#
#     def train_forward(self, batch):
#         data_jigsaw, _ = batch
#
#         """
#         ======================jigsaw===========================
#         """
#         # compute output
#         if self.hparams.is_DDP:
#             images_gather, permute, bs_all = self._batch_gather_ddp(data_jigsaw)
#         else:
#             images_gather, permute, bs_all = self._batch_gather(data_jigsaw)
#         # compute features
#         q = self.backbone(images_gather)
#         q = F.interpolate(q, size=(8,8), mode='bilinear', align_corners=False)
#         q = self.ada_avg_pool2d(q)
#
#         if self.hparams.is_DDP:
#             q_gather = concat_all_gather(q)
#         else:
#             q_gather = q
#
#         n, c, h, w = q_gather.shape
#         c1, c2 = q_gather.split([1, 1], dim=2)
#         f1, f2 = c1.split([1, 1], dim=3)
#         f3, f4 = c2.split([1, 1], dim=3)
#         q_gather = torch.cat([f1, f2, f3, f4], dim=0)
#         q_gather = q_gather.view(n * 4, -1)
#
#         # clustering branch
#         # for way-clustering
#         label_clu = torch.LongTensor(
#             [j for j in range(self.hparams.way)] * (self.hparams.train_shot + self.hparams.num_query) * 4)
#         label_clu = label_clu[permute]
#         # for image-clustering
#         # label_clu = permute % bs_all
#         q_clu = self.fc_clu(q_gather)
#         q_clu = nn.functional.normalize(q_clu, dim=1)
#
#         # location branch
#         label_loc = torch.LongTensor([0] * bs_all + [1] * bs_all + [2] * bs_all + [3] * bs_all).cuda()
#         label_loc = label_loc[permute]
#         q_loc = self.fc_loc(q_gather)
#
#         return q_clu, label_clu, q_loc, label_loc
#
#     def val_test_forward(self, batch, batch_size, way, shot):
#         num_support_samples = way * shot
#         data, _ = batch
#         data = self.backbone(data)
#         data = self.ada_max_pool2d(data) # 不知道为什么 效果好
#         data = data.reshape([batch_size, -1] + list(data.shape[-3:]))
#         data_support = data[:, :num_support_samples]
#         data_query = data[:, num_support_samples:]
#         logits = self.val_test_classifier(data_query, data_support, way, shot)
#         return logits
#
#     def training_step(self, batch, batch_idx):
#         q_clu, label_clu, q_loc, label_loc = self.train_forward(batch)
#         """
#         ==================Jigsaw=======================
#         """
#         loss_clu = self.criterion_clu(q_clu, label_clu)
#         log_loss_clu = self.CLU_loss(loss_clu)
#         self.log("train/loss_clu", log_loss_clu)
#
#         loss_loc = self.criterion_loc(q_loc, label_loc)
#         log_loss_loc = self.LOC_loss(loss_loc)
#         self.log("train/loss_clu", log_loss_loc)
#
#         log_accuracy_loc = self.LOC_acc(q_loc, label_loc)
#         self.log("train/loc_acc", log_accuracy_loc)
#
#
#         loss = self.hparams.loss_clu_weight * loss_clu + self.hparams.loss_loc_weight * loss_loc
#         self.log("train/loss_total", loss)
#
#         return loss
#
#     def set_metrics(self):
#         """Set logging metrics."""
#         self.CLU_loss = AverageMeter()
#         self.LOC_loss = AverageMeter()
#         self.LOC_acc = Accuracy()
#         for split in ["val", "test"]:
#             setattr(self, f"{split}_loss", AverageMeter())
#             setattr(self, f"{split}_acc", Accuracy())
#
#
#
#
# def concat_all_gather(tensor):
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     import diffdist
#     tensors_gather = diffdist.functional.all_gather(tensors_gather, tensor, next_backprop=None, inplace=True)
#
#     output = torch.cat(tensors_gather, dim=0)
#     return output
#
#
# def get_model():
#     return Jigsaw
