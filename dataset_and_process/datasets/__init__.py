from importlib import import_module

def get_dataset(dataset_name: str):
    r"""Import a dataset given the name of file that contains it.

    Args:
        dataset_name: the name file that contains wanted dataset
    """
    ### If the name is specified in relative terms, then the package argument must be set to the name of the package
    dataset_module = import_module("." + dataset_name, package="dataset_and_process.datasets")
    ### get return_class function
    return_class =  getattr(dataset_module, "return_class")
    return return_class()