from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.trainer.trainer import Trainer
from modules import get_module, BaseFewShotModule
from dataset_and_process import get_datamodule, FewShotDataModule
from pytorch_lightning.core.lightning import LightningModule
from callbacks import RefinedSaverCallback
import torch
import numpy as np
import json
import os
import utils
from pytorch_lightning.utilities.seed import seed_everything
from architectures import get_classifier, get_backbone
def simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1),beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1),beta)
    return x_pos+x_neg
class Few_Shot_CLI(LightningCLI):
    """Add testing, model specifying and loading proccess into LightningCLI.
           Add four config parameters:
             --is_test: determine the mode
             --model_name: The few-shot model name. For example, PN.
             --load_pretrained: whether to load pretrained model.
             --pre_trained_path: The path of pretrained model.
             --load_backbone_only: whether to only load the backbone.
             --num_test: The number of processes of implementing testing.
                         The average accuracy and 95% confidence interval across
                         all repeated processes will be calculated.
    """
    def __init__(self,**kwargs) -> None:
        """
        Args:
            kwargs: Original parameters of LightningCLI
        """
        super().__init__(**kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            'is_test',
            type=bool,
            default=False,
            help="whether in testing only mode"
        )
        parser.add_argument(
            'model_name',
            type=str,
            default="PN",
            help="The model name to train on.\
                  It should match the file name that contains the model."
        )
        parser.add_argument(
            'datamodule_name',
            type=str,
            default="few_shot_datamodule",
            help="The datamodule name.\
                  It should match the file name that contains the module."
        )
        parser.add_argument(
            'load_pretrained',
            type=bool,
            default=False,
            help="whether load pretrained model.\
                 This is is different from resume_from_checkpoint\
                 that loads everything from a breakpoint."
        )
        parser.add_argument(
            'pre_trained_path',
            type=str,
            default="",
            help="The path of pretrained model. For testing only."
        )
        parser.add_argument(
            'load_backbone_only',
            type=bool,
            default=False,
            help="whether only load the backbone."
        )
        parser.add_argument(
            'num_test',
            type=int,
            default=5,
            help=r"The number of processes of implementing testing.\
                  The average accuracy and 95% confidence interval across\
                  all repeated processes will be calculated."
        )
    def parse_arguments(self) -> None:
        """Rewrite for skipping check."""
        self.config = self.parser.parse_args(_skip_check = True)
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Rewrite for skipping check."""
        log_dir = trainer.log_dir or trainer.default_root_dir
        config_path = os.path.join(log_dir, self.config_filename)
        self.parser.save(self.config, config_path, skip_none=False, skip_check=True)
    def before_instantiate_classes(self) -> None:
        """get the configured model"""
        self.model_class = get_module(self.config["model_name"])
        self.datamodule_class = get_datamodule(self.config["datamodule_name"])

    def before_fit(self):
        """Load pretrained model."""
        string = "cuda:1"
        if self.config["load_pretrained"]:
            state = torch.load(self.config["pre_trained_path"], map_location=string)["state_dict"]
            if self.config["load_backbone_only"]:
                state = utils.preserve_key(state, "backbone")
                self.model.backbone.load_state_dict(state)
            else:
                self.model.load_state_dict(state)

    def fit(self):
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""
        if self.config["is_test"]:
            pass
        else:
            self.trainer.fit(**self.fit_kwargs)
    def after_fit(self):
        """Runs testing and logs the results"""
        seed_everything(10)
        if self.config["is_test"]:
            acc_list = []
            for _ in range(self.config["num_test"]):
                result=self.trainer.test(self.model, datamodule=self.datamodule)
                acc_list.append(result[0]['test/acc']*100)
            acc_list = np.array(acc_list)
            mean = np.mean(acc_list)
            confidence_interval = np.std(acc_list)*1.96
            with open(os.path.join(self.trainer.log_dir, "test_result.json"), 'w') as f:
                json.dump({'mean':mean, "confidence interval": confidence_interval}, f)
            print(mean)
            print(confidence_interval)
        else:
            pass

if __name__ == '__main__':
    cli = Few_Shot_CLI(
        model_class= BaseFewShotModule, 
        datamodule_class = None, 
        # seed_everything_default=1234,
        save_config_callback = RefinedSaverCallback
    )