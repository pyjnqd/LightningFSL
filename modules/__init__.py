from importlib import import_module
from .base_module import BaseFewShotModule
def get_module(module_name):
    model_module = import_module("." + module_name, package="modules")
    get_model_hook = getattr(model_module, "get_model")
    return get_model_hook()