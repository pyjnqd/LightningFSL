from .classifier.proto_head import PN_head
from .classifier.CC_head import CC_head
from .classifier.SOC import SOC
from .classifier.LR import Logistic_Regression
from .classifier.metaopt import MetaoptHead
from .classifier.KNN import KNN_head
from .classifier.matchnet_head import MN_head


from importlib import import_module

def get_backbone(architecture_name, *args, **kwargs):
    architecture_module = import_module("." + architecture_name, package="architectures.feature_extractor")
    create_model = getattr(architecture_module, "create_model")
    return create_model(*args, **kwargs)

def get_classifier(architecture_name,**kwargs):
    architecture_module = import_module("." + architecture_name, package="architectures.classifier")
    create_model = getattr(architecture_module, "create_model")
    return create_model(**kwargs)