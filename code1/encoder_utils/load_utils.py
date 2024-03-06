import torch
from collections import OrderedDict



def get_child_state_dict(state_dict, key):
    return {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("{}.".format(key))}

            
def load_transmvsnet_checkpoint(model_mvs, ckpt_path, gmflow_n_blocks=6):
    mvs_ckpt = torch.load(ckpt_path)
    mvs_weights = mvs_ckpt['model'] if 'model' in mvs_ckpt else mvs_ckpt
    
    model_mvs.load_state_dict(mvs_weights, strict=True)
    
    # mvs_weights_updated = OrderedDict()
    # for k, v in mvs_weights.items():
    #     keep_key = True
    #     for idx in range(gmflow_n_blocks, 6):
    #         if k.startswith("transformer.layers.%d" % idx):
    #             keep_key = False
    #             break
    #     if k.startswith('upsampler'):  # remove the gmflow upsampler
    #         keep_key = False
    #     if k.startswith('feature_flow_attn'):  # do not need the refine self-attention anymore
    #         keep_key = False
    #     if keep_key:
    #         mvs_weights_updated[k] = v

