from sacred import Experiment
import yaml

ex = Experiment("Test_Only", save_git_info=False)


@ex.config
def config():
    config_dict = {}

    '''
    # ======================= cli attached config parameters ==========================
    #   This part of config is used in CLI execute flow. In principle,
    #  they only are used in CLI if they aren't transfered to inner module.
    '''
    # if training from scratch, set to False. Otherwise set to True if wanna train from a pretrained one.
    config_dict["load_pretrained"] = False
    # if training, set to False
    config_dict["is_test"] = True

    if config_dict["is_test"]:
        # if testing, specify the total rounds of testing. Default: 5
        config_dict["num_test"] = 5
        config_dict["load_pretrained"] = True
    if config_dict["load_pretrained"]:
        config_dict["pre_trained_path"] = "/home/wuhao/workspace/LightningFSL/resume/protonet.ckpt"
        config_dict["load_backbone_only"] = True

    # Specify the module name, which should match the name of file
    # that contains the LightningModule or LightningDataModule
    config_dict["model_name"] = "test_only"
    config_dict["datamodule_name"] = "few_shot_datamodule"

    '''
    # ====================== Component Configuration =========================
        Trainer, Model and Data relevant configurations
        This three parts will be saved in yaml file and used in Lightning pipeline.
    '''

    trainer = {}
    data = {}
    model = {}
    config_dict["trainer"] = trainer
    config_dict["data"] = data
    config_dict["model"] = model

    # ---------------------- Trainer Configuration -------------------------

    # local vars for subsequent trainer config
    multi_gpu = True
    if config_dict["is_test"]:
        multi_gpu = False
    seed = 10
    log_dir = "./results/"
    exp_name = "Test_Only/"

    # ------- Trainer Configurations
    # debugging mode, for example doing val epoch after achieving one training epoch
    trainer["fast_dev_run"] = False
    if multi_gpu:
        trainer["accelerator"] = "ddp"
        trainer["sync_batchnorm"] = True
        trainer["gpus"] = [0, 1]
        trainer["plugins"] = [{"class_path": "plugins.modified_DDPPlugin"}]
    else:
        trainer["accelerator"] = None
        trainer["gpus"] = [0]
        trainer["sync_batchnorm"] = False  # Default:False

    trainer["resume_from_checkpoint"] = None
    trainer["max_epochs"] = 60

    # potential functionalities added to the trainer.
    trainer["callbacks"] = [
        {"class_path": "pytorch_lightning.callbacks.LearningRateMonitor",
         "init_args": {"logging_interval": "step"}
         },
        {"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
         "init_args": {"verbose": True, "save_last": True, "monitor": "val/acc", "mode": "max"}
         },
        {"class_path": "callbacks.SetSeedCallback",
         "init_args": {"seed": seed, "is_DDP": multi_gpu}
         }
    ]
    trainer["logger"] = {"class_path": "pytorch_lightning.loggers.TensorBoardLogger",
                         "init_args": {"save_dir": log_dir, "name": exp_name}
                         }
    trainer["replace_sampler_ddp"] = False  # Default: True. If set False, need to implement own sampler
    # trainer["profiler"] = "pytorch"

    # ----------- Data and Model Configuration
    # local vars
    num_gpus = trainer["gpus"] if isinstance(trainer["gpus"], int) else len(trainer["gpus"])

    per_gpu_train_batchsize = 2
    per_gpu_val_batchsize = 2
    per_gpu_test_batchsize = 8  # LR KNN classifier size 1

    train_shot = 5
    val_shot = 5
    test_shot = 5
    way = 5
    num_query = 15

    # -------Data

    # dataset
    # /mnt/hdd1/wuhao/mini_imagenet/images_imagefolder/
    # /dev/shm/wuhao/mini/images_imagefolder
    data["train_dataset_name"] = "miniImageNet"
    data["train_data_root"] = "/mnt/hdd1/wuhao/mini_imagenet/images_imagefolder/"
    data["val_dataset_name"] = "miniImageNet"
    data["val_data_root"] = "/mnt/hdd1/wuhao/mini_imagenet/images_imagefolder/"
    data["test_dataset_name"] = "pickle_dataset"
    data["test_data_root"] = "/home/wuhao/workspace/LightningFSL/pickles/PN_resnet12_on_mini_test.plk"
    # determine whether meta-learning.
    data["is_meta"] = True
    # loader num_workers
    data["train_num_workers"] = 12
    data["val_num_workers"] = 12
    # the number of tasks
    data["train_num_task_per_epoch"] = 1000
    data["val_num_task"] = 1200
    data["test_num_task"] = 2000
    # way shot query
    data["way"] = way
    data["num_query"] = num_query
    data["train_shot"] = train_shot
    data["test_shot"] = test_shot
    data["val_shot"] = val_shot
    # batch_size
    data["train_batchsize"] = num_gpus * per_gpu_train_batchsize
    data["is_FSL_val"] = True
    if data["is_FSL_val"]:
        data["val_batchsize"] = num_gpus * per_gpu_val_batchsize
        data["test_batchsize"] = num_gpus * per_gpu_test_batchsize
    else:
        data["val_batchsize"] = 1024
        data["test_batchsize"] = 1024

    # Other
    data["is_DDP"] = True if multi_gpu else False
    data["drop_last"] = False

    # ------- Model(LightningModule includes Model, Optimizer and so on, different from Pytorch)

    model["backbone_name"] = "resnet12"
    # the initial learning rate
    # lr is relevant to batch-size commonly
    model["lr"] = 0.1 * data["train_batchsize"] / 4
    # way shot query
    model["way"] = way
    model["train_shot"] = train_shot
    model["val_shot"] = val_shot
    model["test_shot"] = test_shot
    model["num_query"] = num_query

    model["train_batch_size_per_gpu"] = per_gpu_train_batchsize
    model["val_batch_size_per_gpu"] = per_gpu_val_batchsize
    model["test_batch_size_per_gpu"] = per_gpu_test_batchsize
    model["weight_decay"] = 5e-4
    # The name of optimization scheduler
    model["decay_scheduler"] = "cosine"
    model["optim_type"] = "sgd"
    # cosine or euclidean
    model["metric"] = "cosine"  # enclidean cosine
    model["scale_cls"] = 10.0


@ex.automain
def main(_config):
    config_dict = _config["config_dict"]
    config_file_path = 'config/config.yaml'
    stream = open(config_file_path, 'w')
    yaml.safe_dump(config_dict, stream=stream, default_flow_style=False)

# --------------------- execute explanation --------------------------- #
# when run this experiment, scared will run config() and put all        #
# local vars into the configuration of this experiment.All of them      #
# can be accessed in main() decorated with @ex.automain. We can save    #
# some critical configurations, that is to say, the whole experiment    #
# configurations transfer to main() automatically in read-only dict     #
# format.                                                               #
# --------------------------------------------------------------------- #

