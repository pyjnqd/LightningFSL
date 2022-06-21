import torch
import re
import collections
# state = torch.load("/home/wuhao/workspace/JigsawClustering/outputs/jigclu_pretrain/model_best.pth.tar")

state2 = torch.load("/home/wuhao/workspace/LightningFSL/results/CL/Jigsaw/version_47/checkpoints/epoch=99-step=24999.ckpt")
keys = list(state2['state_dict'].keys())
state = torch.load("/home/wuhao/workspace/FEAT/checkpoints/MiniImageNet-ProtoNet-Res12-05w01s15q-DIS/10_0.2_lr0.0001mul10_step_T120.0T21.0_b0.0_bsz080-NoAug/protonet-1-shot.pth")
#
# for key in keys:
#     state['state_dict'][key] = state2['state_dict'][key]
state_dict = {}
i = 0
for k in state['params']:
    if k.startswith('encoder'):
        state_dict[keys[i]] = state['params'][k]
        i += 1
print(i)
        # state_dict['backbone.' + k[len('encoder.'):]] = state2['state_dict'][k]
        #state_dict[keys[i]] = state2['state_dict'][k]

        # state_dict["classifier.scale_cls"] = 10.0
n_state = {'state_dict':state_dict}
# for i in state:
#     if i != 'state_dict':
#         n_state[i] = state[i]
#     else:
#         n_state['state_dict'] = collections.OrderedDict()
#         for j in state['state_dict']:
#             if not j.startswith('t_backbone'):
#                 n_state['state_dict'][j] = state['state_dict'][j]

torch.save(n_state, "/home/wuhao/workspace/FEAT/checkpoints/MiniImageNet-ProtoNet-Res12-05w01s15q-DIS/10_0.2_lr0.0001mul10_step_T120.0T21.0_b0.0_bsz080-NoAug/protonet-1-shot-new.pth")



