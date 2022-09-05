import torch
import re
import collections

state = torch.load("/home/wuhao/workspace/LightningFSL/resume/metaopt-res12-new.ckpt", map_location="cuda:0")
keys = list(state['state_dict'].keys())
print(keys)
exit()
state_dict = {}
for k in state['state_dict']:
    if k.startswith("backbone.layer") and not k.endswith('tracked'):
        state_dict['backbone.layer'+k[14:15]+k[17:] ] = state['state_dict'][k]
#     if k.startswith('block') and not k.endswith('tracked'):
#         state_dict['backbone' + k[len('block3'):]] = state['state'][k]

n_state = {}
n_state = {'state_dict': state_dict}
torch.save(n_state, "/home/wuhao/workspace/LightningFSL/resume/metaopt-res12-new.ckpt")
exit()
keys = list(state2['state_dict'].keys())
print(keys)
exit()
state = torch.load("/home/wuhao/workspace/LightningFSL/results/ProtoNet/Res12-pre.pth")
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



