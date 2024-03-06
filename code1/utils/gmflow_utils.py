import torch
from collections import OrderedDict



def get_child_state_dict(state_dict, key):
    return {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("{}.".format(key))}


def load_gmflow_checkpoint(model_enc, ckpt_path, gmflow_n_blocks=6):
    gmflow_ckpt = torch.load(ckpt_path)
    gmflow_weights = gmflow_ckpt['model'] if 'model' in gmflow_ckpt else gmflow_ckpt
    gmflow_weights_updated = OrderedDict()
    for k, v in gmflow_weights.items():
        keep_key = True
        for idx in range(gmflow_n_blocks, 6):
            if k.startswith("transformer.layers.%d" % idx):
                keep_key = False
                break
        if k.startswith('upsampler'):  # remove the gmflow upsampler
            keep_key = False
        if k.startswith('feature_flow_attn'):  # do not need the refine self-attention anymore
            keep_key = False
        if keep_key:
            gmflow_weights_updated[k] = v

    for name, child in model_enc.named_children():
        if name != "featup_net":  # our upsample is different from gmflow's
            child_state_dict = get_child_state_dict(gmflow_weights_updated, name)
            child.load_state_dict(child_state_dict, strict=True)