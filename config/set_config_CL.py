"""
Training MoCo---Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020,
or
Exampler---What makes instance discrimination good for transfer learning? ICLR 2021.
"""


from sacred import Experiment
import yaml

ex = Experiment("CL", save_git_info=False)

@ex.config
def config():
    config_dict = {}

    #if training, set to False
    config_dict["load_pretrained"] = False
    #if training, set to False
    config_dict["is_test"] = True
    if config_dict["is_test"]:
        #if testing, specify the total rounds of testing. Default: 5
        config_dict["num_test"] = 5
        config_dict["load_pretrained"] = True
        #specify pretrained path for testing.
    if config_dict["load_pretrained"]:
        config_dict["pre_trained_path"] = "/home/wuhao/workspace/LightningFSL/results/CL/PN+CL+DIS/version_25/checkpoints/epoch=90-step=15196.ckpt"
        #only load the backbone.
        config_dict["load_backbone_only"] = False

    #Specify the model name, which should match the name of file
    #that contains the LightningModule
    config_dict["model_name"] = "CL"
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
    exp_name = "CL/PN+CL+DIS"
    
    #Three components of a Lightning Running System
    trainer = {}
    data = {}
    model = {}


    ################trainer configuration###########################


    ###important###

    #debugging mode
    trainer["fast_dev_run"] = False

    if multi_gpu:
        trainer["accelerator"] = "ddp"
        trainer["sync_batchnorm"] = False
        trainer["gpus"] = [1,2,3]
        trainer["plugins"] = [{"class_path": "plugins.modified_DDPPlugin"}]
    else:
        trainer["accelerator"] = None
        trainer["gpus"] = [2]
        trainer["sync_batchnorm"] = False
    
    # whether resume from a given checkpoint file
    trainer["resume_from_checkpoint"] = None #"/home/wuhao/workspace/LightningFSL/results/CL/moco/version_2/checkpoints/epoch=62-step=38999.ckpt"

    # The maximum epochs to run
    trainer["max_epochs"] = 100

    
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

    

    ##################shared model and datamodule configuration###########################

    #important
    per_gpu_train_batchsize = 2
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
    
    data["train_dataset_name"] = "miniImageNet_mixedwithcontra"

    data["train_data_root"] = "/dev/shm/wuhao/images_imagefolder"

    data["val_test_dataset_name"] = "miniImageNet"

    data["val_test_data_root"] = "/dev/shm/wuhao/images_imagefolder"



    data["train_num_workers"] = 8
    #the number of tasks
    data["val_num_task"] = 1200
    data["test_num_task"] = 2000
    
    
    #less important
    data["num_gpus"] = num_gpus
    data["train_batchsize"] = num_gpus*per_gpu_train_batchsize
    data["is_FSL_val"] = True # update for new datamodule
    if data["is_FSL_val"]:
        data["val_batchsize"] = num_gpus * per_gpu_val_batchsize
        data["test_batchsize"] = num_gpus * per_gpu_test_batchsize
    else:
        data["val_batchsize"] = 1024
        data["test_batchsize"] = 1024
    data["test_shot"] = test_shot
    data["train_shot"] = train_shot
    data["val_num_workers"] = 8
    data["is_DDP"] = True if multi_gpu else False
    data["way"] = way
    data["val_shot"] = val_shot
    data["num_query"] = num_query
    data["drop_last"] = False
    data["is_meta"] = True

    ##################model configuration###########################

    #important

    #The name of feature extractor, which should match the name of file
    #that contains the model.
    model["backbone_name"] = "resnet12"
    model["beta"] = 2.0

    #the initial learning rate
    model["lr"] = 0.1 * data["train_batchsize"]/4


    #less important
    model["momemtum"] = 0.999
    model["temparature"] = 0.07 # Moco temparature
    model["dis_temparature"] = 4.0 # dostill temparature
    model["queue_len"] = 65400
    model["mlp_dim"] = 128
    
    model["is_DDP"] = multi_gpu
    model['is_Exemplar'] = True
    model['is_Distill'] = True
    if model['is_Distill']:
        model['teacher_path'] = "/home/wuhao/workspace/LightningFSL/results/CL/Exemplar/version_1/checkpoints/epoch=1108-step=166349.ckpt"
        model['teacher_backbone_name'] = 'resnet12'
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

    

    config_dict["trainer"] = trainer
    config_dict["data"] = data
    config_dict["model"] = model



@ex.automain
def main(_config):
    config_dict = _config["config_dict"]
    file_ = 'config/config.yaml'
    stream = open(file_, 'w')
    yaml.safe_dump(config_dict, stream=stream,default_flow_style=False)

    
