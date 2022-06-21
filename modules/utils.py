from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR
from torch.optim import Adam, SGD, AdamW
import math



def epoch_wrapup(pl_module: LightningModule, mode: str):
    r"""On the end of each epoch, log information of the whole
        epoch and reset all metrics.
    
    Args:
        pl_module: An instance of LightningModule.
        mode: The current mode (train, val or test).
    """
    assert mode in ["train", "val", "test"]
    value = getattr(pl_module, f"{mode}_loss").compute()
    if mode == 'train':
        pl_module.log(f"{mode}/loss_epoch", value)
    getattr(pl_module, f"{mode}_loss").reset()
    value = getattr(pl_module, f"{mode}_acc").compute()
    if mode == 'train':
        pl_module.log(f"{mode}/acc_epoch", value)
    getattr(pl_module, f"{mode}_acc").reset()

def set_schedule(pl_module: object) -> object:
    r"""Set the optimizer and scheduler for training.

    Supported optimizer:
        Adam and SGD
    Supported scheduler:
        cosine scheduler and decaying on specified epochs

    Args:
        pl_module: An instance of LightningModule.
    """
    lr = pl_module.hparams.lr
    wd = pl_module.hparams.weight_decay
    decay_scheduler = pl_module.hparams.decay_scheduler
    optim_type = pl_module.hparams.optim_type
    warm_up = pl_module.hparams.warm_up


    if optim_type == "adam":
        optimizer = Adam(pl_module.parameters(),
                                    weight_decay=wd, lr=lr)
    elif optim_type == "sgd":
        optimizer = SGD(pl_module.parameters(),
                                    momentum=0.9, nesterov=True,
                                    weight_decay=wd, lr=lr)
    elif optim_type == "adamw":
        optimizer = AdamW(pl_module.parameters(),
                                    weight_decay=wd, lr=lr)
    else:
        raise RuntimeError("optim_type not supported.\
                            Try to implement your own optimizer.")

    # decay_scheduler
    if decay_scheduler == "cosine":
        if pl_module.trainer.max_steps is None:
            length_epoch = len(pl_module.trainer.datamodule.train_dataloader())
            max_steps = length_epoch * pl_module.trainer.max_epochs
            print(f"length_epoch:{length_epoch}")
            print(f"max_epochs:{pl_module.trainer.max_epochs}")
        else:
            max_steps = pl_module.trainer.max_steps
        # warm_up
        if warm_up != 0 and warm_up is not None:
            warm_up_steps = length_epoch * warm_up
            warm_up_with_cosine_lr = lambda step: step / warm_up_steps if step <= warm_up_steps else 0.5 * (
                        math.cos((step - warm_up_steps) / (max_steps - warm_up_steps) * math.pi) + 1)
            scheduler = LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        else:
            scheduler = {'scheduler': CosineAnnealingLR(optimizer, max_steps),
                         'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    elif decay_scheduler == "specified_epochs":
        decay_epochs = pl_module.hparams.decay_epochs
        decay_power = pl_module.hparams.decay_power
        assert decay_epochs is not None and decay_power is not None
        # warm_up
        if warm_up != 0 and warm_up is not None:
            warm_up_with_multistep_lr = lambda \
                epoch: epoch / warm_up if epoch <= warm_up else 0.1 ** len([m for m in decay_epochs if m <= epoch])
            scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr), 'interval': 'epoch'}
        else:
            scheduler = {'scheduler':
                         MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_power),
                         'interval': 'epoch'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    elif decay_scheduler is None:
        return optimizer
    else:
        raise RuntimeError("decay scheduler not supported.\
                            Try to implement your own scheduler.")






    
        