from architectures import get_backbone
from dataset_and_process.datasets import get_dataset
from torch.utils.data import DataLoader
import collections
import torch.nn.functional as F
import pickle
import os
import torch
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import clip

def simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1),beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1),beta)
    return x_pos+x_neg






print(torch.cuda.device_count())
# print(torch.cuda.device_count())
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.device_count())
def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
string = "cuda:3"
device = torch.device(string)
def main(dataset):
    transform = "I"
    # dataset = "dog_cat"
    # dataset = "man_woman"
    
    # dataset = "mini_train"
    dataset = "mini_test"
    # dataset = "cub"
    # dataset = "dtd"
    # dataset = "traffic"
    # dataset = "CropD"
    # dataset = "ISIC"
    # dataset = "ESAT"
    # dataset = "Sketch"
    # dataset = "QuickD"
    # dataset = "Fungi"

    # dataset = "aircraft"
    # dataset = "omniglot"
    # dataset = "vggflower"
# 

    # dataset = "coco"

    # dataset = "ChestX"

    # dataset = "clipart"
    # dataset = "infograph"
    # dataset = "real"
    # dataset = "painting"

    # path = f"miniPN_on{dataset}"
    # path = f"miniCC_on{dataset}"
    # path = f"test2"
    # path = f"clip_on{dataset}"
    # path = f"Res50ImageNet_on{dataset}"
    path = f"metaopt_resnet12_on_{dataset}"
    # path = f"LFT_{dataset}"
    # path = f"ZN_{dataset}"
    print(torch.cuda.device_count())
    # model = get_backbone('resnet10').to(device)
    # model = get_backbone('res18_url').to(device)
    model = get_backbone('resnet12').to(device)
    # model = get_backbone('res2net12').to(device)
    
    # model = get_backbone('res12_Metaopt').to(device)
    # model = get_backbone('conv-4').to(device)
    # model = get_backbone('WRN_28_10').to(device)
    # model = get_backbone('resnet50').to(device)
    # model = get_backbone('se_net').to(device)
    # model.load_from(np.load(f"../pretrained_model/BiT-M-R50x1.npz"))
    # model = get_backbone('WRN_28_10').to(device)
    # model = get_backbone('densenet').to(device)
    # model = get_backbone('swin_transformer').to(device)
    # model = get_backbone('dino_ViT').to(device)
    # model = get_backbone('mae').to(device)
    # model, _ = clip.load("ViT-B/32", device=device)
    
    #temporary
    # state = torch.load("../results/DeepBDC/pretrain/version_0/checkpoints/last.ckpt", map_location=string)["state_dict"]
    #always in use
    # state = torch.load("../results/ProtoNet/noleakyrelu&norm/version_0/checkpoints/epoch=53-step=13499.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/nonorm_PN/version_2/checkpoints/epoch=53-step=26999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../results/meta_baseline_pretrain/first_ex/version_0/checkpoints/epoch=99-step=29999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../pretrained_model/resnet50_official.pth")

    # state = torch.load("../pretrained_model/BYOL_R50.pth", map_location=string)

    # state = torch.load("../pretrained_model/moco_v2_800ep_pretrain.pth.tar", map_location=string)["state_dict"]

    # state = torch.load("../results/ProtoNet/version_11/checkpoints/epoch=52-step=26499.ckpt", map_location=string)["state_dict"]
    
    # state = torch.load("../ICML_results/nonormCC/version_0/checkpoints/epoch=55-step=16799.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../results/CC/noleakynorm/version_1/checkpoints/epoch=51-step=15599.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../results/CC/noleakynorm_tiered/version_2/checkpoints/epoch=54-step=192829.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../pretrained_model/wrn_pre.pth", map_location=string)["params"]
    # state = torch.load("../results/CC/first_ex/version_2/checkpoints/epoch=58-step=17699.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/SEnetCC/version_8/checkpoints/epoch=48-step=14699.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/inaturalist-CC/checkpoints/last.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/Imagenet-PN/version_0/checkpoints/last.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/CC/Imagenet/version_2/checkpoints/last.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/res12-Metaopt/version_5/checkpoints/epoch=59-step=29999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/Conv4-PN/version_0/checkpoints/epoch=59-step=29999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/meta_baseline_finetune/version_8/checkpoints/epoch=2-step=299.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/CC/Fungi_split/version_0/checkpoints/last.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/tieredCCEMDfinetune/version_7/checkpoints/epoch=15-step=3199.ckpt", map_location=string)["state_dict"]
    
    # state = torch.load("../ICML_results/PN_seed/seed=10/version_1/checkpoints/epoch=53-step=26999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/PN_seed/seed=20/version_0/checkpoints/epoch=56-step=28499.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/PN_seed/seed=30/version_0/checkpoints/epoch=58-step=29499.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/PN_seed/seed=40/version_0/checkpoints/epoch=54-step=27499.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/PN_seed/seed=50/version_0/checkpoints/epoch=53-step=26999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../results/ProtoNet/test1gpu/version_0/checkpoints/epoch=59-step=29999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/rebuttal/PNrelu+ZN/version_1/checkpoints/epoch=54-step=13749.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/rebuttal/MetaOPT_leaky/version_0/checkpoints/epoch=52-step=13249.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/rebuttal/MetaOPT_relu_same/version_1/checkpoints/epoch=27-step=55999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../ICML_results/rebuttal/MetaOPT_leakyrelu_same/version_0/checkpoints/epoch=41-step=83999.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../results/ProtoNet/noleakynorm_tiered/version_1/checkpoints/epoch=56-step=28499.ckpt", map_location='cuda:0')["state_dict"]
    # state = torch.load("../results/ProtoNet/github_one/PN.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../results/domainnet_train_CE/CCres12/version_1/checkpoints/epoch=45-step=22631.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../pretrained_model/BiT-M-R50x1", map_location=string)["state_dict"]
    # state = torch.load("../results/ProtoNet/mininormal_noleakynorm/version_0/checkpoints/epoch=59-step=29999.ckpt", map_location='cuda:0')["state_dict"]
    # state = torch.load("../results/CC/noleakynorm/version_1/checkpoints/epoch=51-step=15599.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../results/ProtoNet/noleakynorm_tiered/version_1/checkpoints/epoch=56-step=28499.ckpt", map_location='cuda:1')["state_dict"]
    # state = torch.load("../results/domainnet_train_CE/CCres12/version_1/checkpoints/epoch=45-step=22631.ckpt", map_location=string)["state_dict"]
    # state = torch.load("../results/CC/rbd_train/version_0/checkpoints/epoch=58-step=17699.ckpt")["state_dict"]
    # state = torch.load("../results/Exemplar/1000epoch/no_sync_BN/version_2/checkpoints/epoch=863-step=129599.ckpt",map_location="cuda:6")["state_dict"]
    # state = torch.load("../results/S2M2/train/version_9/checkpoints/epoch=574-step=344999.ckpt")["state_dict"]
    # state = torch.load("../results/MoCo/1000epoch/version_7/checkpoints/epoch=928-step=139349.ckpt")["state_dict"]
    # state = torch.load("/home/wuhao/workspace/LightningFSL/resume/S2M2_R_WRN_28_10-new.pth",map_location=string)["state_dict"]
    # state = torch.load("../results/S2M2/train/version_9/checkpoints/epoch=574-step=344999.ckpt",map_location=string)["state_dict"]
    # state = torch.load("../meta_learning_framework/results/miniImageNet/res12_PN_prob_crop/05_21_10_37_CC_save_all_3crop_alpha0.5_seed10/encoder_net_epoch59.best")["network"]
    # state = torch.load("../meta_learning_framework/results/miniImageNet/res12_PN_prob_crop/04_30_15_43_finetune_save_all_3crop_alpha0.5/encoder_net_epoch16.best")["network"]
    # state = torch.load("../meta_learning_framework/results/miniImageNet/res12_Moco/03_15_10_13_lr0.1_1000epoch_remove_same_class/encoder_net_epoch1000")["network"]
    # state = torch.load("../pretrained_model/feat-1-shot.pth")["params"]
    # state = torch.load("../pretrained_model/deepemd_trained_model/miniimagenet/sampling/max_acc.pth")["params"]
    # state = torch.load("../pretrained_model/miniImagenet_neg_cosine_5_shot.tar")["state"]
    # state = torch.load("../pretrained_model/simple_shot/results/inatural/softmax/densenet121/model_best.pth.tar", map_location=string)["state_dict"]
    # state = torch.load("../pretrained_model/swav_800ep_pretrain.pth.tar.1", map_location=string)
    # state = torch.load("../pretrained_model/multi_cub_lft_relationnet_softmax/400.tar", map_location=string)["model_state"]
    state = torch.load("/home/wuhao/workspace/LightningFSL/resume/metaopt-res12-new.ckpt", map_location=string)[
        "state_dict"]
    # state = torch.load("../pretrained_model/url/model_best.pth.tar", map_location=string)["state_dict"]
    # state = torch.load("../pretrained_model/IER/S:resnet12_T:resnet12_miniImageNet_kd_r:1.0_a:0_b:0_trans_A_tag_INV_EQ_DISTILL_1/model_smart-pond-8.pth", map_location=string)["model"]
    # state = torch.load("../pretrained_model/swin_base_patch4_window7_224.pth", map_location=string)["model"]
    # state = torch.load("../pretrained_model/dino_deitsmall8_pretrain.pth", map_location=string)
    # state = torch.load("../pretrained_model/pretrain_mae_base_0.75_400e_ft_100e.pth", map_location=string)["model"]
    # for key, value in state.items():
    #     print(key)
    # import pdb
    # pdb.set_trace()
    state = utils.preserve_key(state, "backbone")
    
    #for IER
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     if "inv_head" in key or "eq_head" in key or "classifier" in key:
    #         state.pop(key)
    #     else:
    #         newkey = key.replace("module.", "")
    #         newkey = newkey.replace("layer1.0", "layer1")
    #         newkey = newkey.replace("layer2.0", "layer2")
    #         newkey = newkey.replace("layer3.0", "layer3")
    #         newkey = newkey.replace("layer4.0", "layer4")
    #         # print(newkey)
    #         state[newkey] = state.pop(key)
    # for key, value in state.items():
    #     print(key)

    # for url
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     if "cls_fn" in key:
    #         state.pop(key)
    #for LFT
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     if "feature" in key and not 'gamma' in key and not 'beta' in key:
    #         newkey = key.replace("feature.", "")
    #         state[newkey] = state.pop(key)
    #     else:
    #         state.pop(key)

    #for moco
    # state = utils.preserve_key(state, "module.encoder_q")
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     if "fc" in key:
    #         state.pop(key)

    #for BYOL and res50-official
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     if "fc" in key:
    #         state.pop(key)

    #for swav
    # state = utils.preserve_key(state, "module")
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     if "projection" in key or "prototype" in key:
    #         state.pop(key)

    #for WRN_pre
    # state = utils.preserve_key(state, "encoder")
    # state_keys = list(state.keys())
    
    #for simple_shot
    # state = utils.preserve_key(state, "module")
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     if "classifier" in key:
    #         state.pop(key)

    # for i, key in enumerate(state_keys):
    #     if ".0" in key:
    #         newkey = key.replace(".0", "")
    #         state[newkey] = state.pop(key)
        # if "backbone_m" in key:
        #     state.pop(key)
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     if "downsample.weight" in key:
    #         newkey = key.replace("downsample.weight", "downsample.0.weight")
    #         state[newkey] = state.pop(key)
        # if "backbone_teacher" in key:
        #     state.pop(key)
        # else:
        #     state.pop(key)
    # return state
    # state_keys = list(state.keys())
    # for i, key in enumerate(state_keys):
    #     print(key)
    model.load_state_dict(state)

    # model = torch.hub.load('facebookresearch/swav:main', 'resnet50').to(device)
    
    # dataset = get_dataset("miniImageNet_sampling")("../BF3S-master/data/mini_imagenet_split/images", "test",9)
    # dataset = get_dataset("miniImageNet_sampling")("../BF3S-master/data/mini_imagenet_split/images", "train",9)
    # dataset = get_dataset("miniImageNet")("../BF3S-master/data/mini_imagenet_split/normal_images", "train")
    # dataset = get_dataset("miniImageNet")("../BF3S-master/data/mini_imagenet_split/images", "test")
    # dataset = get_dataset("tieredImageNet")("../CrossDomainFewShot-master/filelists/tiered_imagenet", "test")
    # dataset = get_dataset("aircraft")("../data/fgvc-aircraft-2013b/data", "test")
    # dataset = get_dataset("coco")("../data/coco2017")
    # dataset = get_dataset("omniglot")("../data/omniglot", False)
    # dataset = get_dataset("vggflower")("../data/vggflowers")
    if dataset == "real":
        dataset = get_dataset("general_dataset")("../data/domainnet/real", "test", transform)
    elif dataset == "infograph":
        dataset = get_dataset("general_dataset")("../data/domainnet/infograph", "test",transform)
    elif dataset == "painting":
        dataset = get_dataset("general_dataset")("../data/domainnet/painting", "test",transform)
    elif dataset == "Sketch":
        dataset = get_dataset("general_dataset")("../data/domainnet/sketch", "test",transform)
    elif dataset == "QuickD":
        dataset = get_dataset("general_dataset")("../data/domainnet/quickdraw", "test",transform)
    elif dataset == "clipart":
        dataset = get_dataset("general_dataset")("../data/domainnet/clipart", "test",transform)
    elif dataset == "mini_test":
        dataset = get_dataset("miniImageNet")("/mnt/hdd1/wuhao/mini_imagenet/images_imagefolder/", "test")
    elif dataset == "mini_train":
        dataset = get_dataset("general_dataset")("../BF3S-master/data/mini_imagenet_split/images/train", "test",transform)
    elif dataset == "cub":
        dataset = get_dataset("general_dataset")("../data/CUB_200_2011/images", "test",transform)
    elif dataset == "dtd":
        dataset = get_dataset("general_dataset")("../data/dtd/images", "test",transform)
    elif dataset == "traffic":
        dataset = get_dataset("general_dataset")("../data/GTSRB/Final_Training/Images", "test",transform)
    elif dataset == "CropD":
        dataset = get_dataset("general_dataset")("../data/cross_domain/plant_disease/train", "test",transform)
    elif dataset == "ESAT":
        dataset = get_dataset("general_dataset")("../data/cross_domain/EuroSAT", "test",transform)
    elif dataset == "Fungi":
        dataset = get_dataset("general_dataset")("../data/fungi/images", "test",transform)
    elif dataset == "ISIC":
        dataset = get_dataset("ISIC")("../data/cross_domain/ISIC2018_Task3_Training_Input/", "../data/cross_domain/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv", "all", transform)
    elif dataset == "aircraft":
        dataset = get_dataset("aircraft")("../data/fgvc-aircraft-2013b/data", "test",transform)
    elif dataset == "omniglot":
        dataset = get_dataset("omniglot")("../data/omniglot", False, transform)
    elif dataset == "vggflower":
        dataset = get_dataset("vggflower")("../data/vggflowers", transform)
    elif dataset == "coco":
        dataset = get_dataset("coco")("../data/coco2017", transform)
    elif dataset == "ChestX":
        dataset = get_dataset("chestX")("../data/cross_domain/chest/Data_Entry_2017.csv", "../data/cross_domain/chest/images/", transform)
    elif dataset == "dog_cat":
        dataset = get_dataset("general_dataset")("../some_image/humandog/dog_cat", "test", transform)
    elif dataset == "man_woman":
        dataset = get_dataset("general_dataset")("../some_image/humandog/man_woman", "test", transform)
    # dataset = get_dataset("ImageNet")("../data/fungi/images", "test")
    # dataset = get_dataset("bit_config")("../BF3S-master/data/mini_imagenet_split/images/test", "test")
    # dataset = get_dataset("chestX")("../data/cross_domain/chest/Data_Entry_2017.csv", "../data/cross_domain/chest/images/")
    # dataset = get_dataset("ISIC")("../data/cross_domain/ISIC2018_Task3_Training_Input/", "../data/cross_domain/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv", "all")
    # dataset = get_dataset("general_dataset")("../CrossDomainFewShot-master/filelists/tiered_imagenet/test", "test")
    # dataset = get_dataset("general_dataset")("../data/fungi/images","test")
    # dataset = get_dataset("miniImageNet_old")("../dataset/miniImageNet", "test", True,7)
    # dataset = get_dataset("miniImageNet_saliency")("../BF3S-master/data/mini_imagenet_split/images", "test", "../pyimgsaliency/test_mbd.plk", True,7)
    dataloader = DataLoader(dataset, 128, shuffle=False, num_workers=8, pin_memory=True)
    model.eval()

    save_dir = "./pickles"
    with torch.no_grad():
        output_dict = collections.defaultdict(list)
        all_data = []
        all_labels = []
        count = 0
        statistics = 0.
        for i, (data, labels) in enumerate(tqdm(dataloader)):
            
            batch_size = data.size(0)
            # data = data.to(device)
            # data = data.permute(0,3,1,2)
            # labels = labels.to(device)

            # import pdb
            # pdb.set_trace()
            # is_patch = False
            data = data.to(device)
            labels = labels.to(device)
            # if data.shape.__len__() == 5:
            #     is_patch = True
            #     num_patch = data.size(1)
            #     data = data.reshape([-1]+list(data.shape[-3:]))
            # print(data.shape)
            # data = model.encode_image(data)
            data = model(data)
            # print(data.shape)
            # if is_patch:
            #     # data = simple_transform(data,1.3)
            #     # data = F.normalize(data, dim=1)
            #     data = F.adaptive_avg_pool2d(data, 1)
            #     data = data.reshape([batch_size, num_patch] + list(data.shape[-3:]))
            #     data = data.permute(0, 2, 1, 3, 4).squeeze(-1)
            #     # data= F.normalize(data, dim=1)
            #     # data = F.adaptive_avg_pool2d(data, 1).squeeze_(-1).squeeze_(-1)
                
            # else:
            # data = simple_transform(data,1.3)
            # data = F.normalize(data, dim=1)
            # data = torch.nn.AvgPool2d(2, stride=2)(data)
            # print(data.shape)
            if data.shape.__len__()==4:
                data = F.adaptive_avg_pool2d(data, 1).squeeze_(-1).squeeze_(-1)
            else:
                pass
            data = F.normalize(data, dim=1) #note
            # data[:,8]=0
            # data[:,56]=0
            # data = data[:, 8].unsqueeze_(1)
            # print(data.shape)
            # statistics += data.mean(0)
            # count+=1
            # print(torch.mean(data))
            # print(torch.max(data))
            # print(torch.std(data))
            # assert data.dim() == 2
            # print(data.shape)
            # data = data.cpu().data.numpy()
            all_data.append(data.cpu())
            all_labels.append(labels.cpu())
        # statistics /=count
        # statistics = statistics.cpu().numpy().tolist()
        # plt.bar(range(640), statistics)
        # plt.savefig("hh1.jpg")


        
        all_data = torch.cat(all_data,dim=0).cpu().data.numpy()
        all_labels = torch.cat(all_labels,dim=0).cpu().data.numpy()
        # save_pickle(save_dir + '/test2.plk', (all_data, all_labels))
        save_pickle(save_dir + '/'+path+ '.plk', (all_data, all_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('argument for testing')
    parser.add_argument('--dataset', type=str, default="mini_test")
    opt = parser.parse_args()
    main(opt.dataset)
    # with open("../data/pickles/COS_7s_features.plk", 'rb') as f:
    #     data = pickle.load(f)
    # import pdb
    # pdb.set_trace()
    # print('hh')