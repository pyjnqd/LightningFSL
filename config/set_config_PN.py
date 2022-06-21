from sacred import Experiment
import yaml
ex = Experiment("ProtoNet", save_git_info=False)

@ex.config
def config():
    config_dict = {}

    # if training, set to False
    config_dict["load_pretrained"] = False
    # if training, set to False
    config_dict["is_test"] = True

    if config_dict["is_test"]:
        # if testing, specify the total rounds of testing. Default: 5
        config_dict["num_test"] = 5
        config_dict["load_pretrained"] = True
        # specify pretrained path for testing.
    if config_dict["load_pretrained"]:
        config_dict["pre_trained_path"] = "/home/wuhao/workspace/FEAT/checkpoints/MiniImageNet-ProtoNet-Res12-05w05s15q-DIS/10_0.2_lr0.0001mul10_step_T120.0T21.0_b0.0_bsz100-NoAug/protonet-5-shot-new.pth"
        # only load the backbone.
        config_dict["load_backbone_only"] = True
        
    #Specify the model name, which should match the name of file
    #that contains the LightningModule
    config_dict["model_name"] = "PN"
    config_dict["datamodule_name"] = "few_shot_datamodule"
 
    

    #whether to use multiple GPUs
    multi_gpu = True
    if config_dict["is_test"]:
        multi_gpu = False
    #The seed
    seed = 10
    config_dict["seed"] = seed

    #The logging dirname: logdir/exp_name/
    log_dir = "./results/"
    exp_name = "ProtoNet/"
    
    #Three components of a Lightning Running System
    trainer = {}
    data = {}
    model = {}


    ################trainer configuration###########################


    ###important###

    #debugging mode, for example doing val epoch after achieving one training epoch 
    trainer["fast_dev_run"] = False

    if multi_gpu:
        trainer["accelerator"] = "ddp"
        trainer["sync_batchnorm"] = True
        trainer["gpus"] = [0,1]
        trainer["plugins"] = [{"class_path": "plugins.modified_DDPPlugin"}]
    else:
        trainer["accelerator"] = None
        trainer["gpus"] = [0]
        trainer["sync_batchnorm"] = False
    
    # whether resume from a given checkpoint file
    trainer["resume_from_checkpoint"] = None

    # The maximum epochs to run
    trainer["max_epochs"] = 80

    # potential functionalities added to the trainer.
    trainer["callbacks"] = [{"class_path": "pytorch_lightning.callbacks.LearningRateMonitor", 
                  "init_args": {"logging_interval": "step"}
                  },
                {"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                  "init_args":{"verbose": True, "save_last": True, "monitor": "val/acc", "mode": "max"}
                },
                {"class_path": "callbacks.SetSeedCallback",
                 "init_args":{"seed": seed, "is_DDP": multi_gpu}
                }]

    ###less important###
    num_gpus = trainer["gpus"] if isinstance(trainer["gpus"], int) else len(trainer["gpus"])
    trainer["logger"] = {"class_path":"pytorch_lightning.loggers.TensorBoardLogger",
                        "init_args": {"save_dir": log_dir,"name": exp_name}
                        }
    trainer["replace_sampler_ddp"] = False
    # trainer["profiler"] = "pytorch"

    

    ##################shared model and datamodule configuration#########

    #important
    per_gpu_train_batchsize = 2 # 每個GPU
    train_shot = 5
    test_shot = 5

    #less important
    per_gpu_val_batchsize = 2
    per_gpu_test_batchsize = 8
    way = 5
    val_shot = 5
    num_query = 15

    ##################datamodule configuration###########################

    #important

    #The name of dataset, which should match the name of file
    #that contains the datamodule.
    data["train_dataset_name"] = "miniImageNet"

    data["train_data_root"] = "/home/wuhao/data/mini_imagenet/images_imagefolder"
    # /dev/shm/wuhao/mini_imagenet/images_imagefolder
    # /home/wuhao/data/mini_imagenet/images_imagefolder
    data["val_test_dataset_name"] = "miniImageNet"

    data["val_test_data_root"] = "/home/wuhao/data/mini_imagenet/images_imagefolder"
    #determine whether meta-learning.
    data["is_meta"] = True
    data["train_num_workers"] = 8
    #the number of tasks
    data["train_num_task_per_epoch"] = 1000
    data["val_num_task"] = 1200
    data["test_num_task"] = 2000
    
    
    #less important
    data["train_batchsize"] = num_gpus * per_gpu_train_batchsize
    data["is_FSL_val"]=True # update for new datamodule
    if data["is_FSL_val"]:
        data["val_batchsize"] = num_gpus*per_gpu_val_batchsize
        data["test_batchsize"] = num_gpus*per_gpu_test_batchsize
    else:
        data["val_batchsize"] = 1024
        data["test_batchsize"] = 1024
    data["test_shot"] = test_shot
    data["train_shot"] = train_shot
    data["val_num_workers"] = 12
    data["is_DDP"] = True if multi_gpu else False
    data["way"] = way
    data["val_shot"] = val_shot
    data["num_query"] = num_query
    data["drop_last"] = False

    ##################model configuration###########################

    #important

    #The name of feature extractor, which should match the name of file
    #that contains the model.
    # if you wnnan use transformer-based backbone, pls quote in the form of vit_pytorch.xxx
    model["backbone_name"] = "resnet12"
    #the initial learning rate
    model["lr"] = 0.1*data["train_batchsize"]/4


    #less important
    model["way"] = way
    model["train_shot"] = train_shot
    model["val_shot"] = val_shot
    model["test_shot"] = test_shot
    model["num_query"] = num_query
    model["train_batch_size_per_gpu"] = per_gpu_train_batchsize
    model["val_batch_size_per_gpu"] = per_gpu_val_batchsize
    model["test_batch_size_per_gpu"] = per_gpu_test_batchsize
    model["weight_decay"] = 5e-4
    #The name of optimization scheduler
    model["decay_scheduler"] = "cosine"
    model["optim_type"] = "sgd"
    #cosine or euclidean
    model["metric"] = "enclidean" # enclidean
    model["scale_cls"] = 0.05
    model["normalize"] = True
    

    config_dict["trainer"] = trainer
    config_dict["data"] = data
    config_dict["model"] = model



@ex.automain
def main(_config):
    config_dict = _config["config_dict"]
    file_ = 'config/config.yaml'
    stream = open(file_, 'w')
    yaml.safe_dump(config_dict, stream=stream, default_flow_style=False)

    