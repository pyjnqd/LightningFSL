from importlib import import_module
from .datamodules.few_shot_datamodule import FewShotDataModule

def get_datamodule(module_name):
    model_module = import_module("." + module_name, package="dataset_and_process.datamodules")
    model_module = getattr(model_module, "get_datamodule")
    return model_module()