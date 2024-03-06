import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


'''
- We provide two different positional encoding methods as shown below.
- You can easily switch different pos-enc in the __init__() function of FMT.
- In our experiments, PositionEncodingSuperGule usually cost more GPU memory.
'''
from .position_encoding import PositionEncodingSuperGule, PositionEncodingSine


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape 
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(EncoderLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, source):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, source, source,
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)


class FMT(nn.Module):
    def __init__(self, config):
        super(FMT, self).__init__()

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = EncoderLayer(config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

        # self.pos_encoding = PositionEncodingSuperGule(config['d_model'])
        self.pos_encoding = PositionEncodingSine(config['d_model'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ref_feature=None, src_feature=None, feat="ref", self_features=None):
        """
        Args:
            ref_feature(torch.Tensor): [N, C, H, W]
            src_feature(torch.Tensor): [N, C, H, W]
        """

        assert ref_feature is not None

        if feat == "ref": # only self attention layer

            assert self.d_model == ref_feature.size(1)
            _, _, H, _ = ref_feature.shape

            ref_feature = einops.rearrange(self.pos_encoding(ref_feature), 'n c h w -> n (h w) c')

            ref_feature_list = []
            for layer, name in zip(self.layers, self.layer_names): # every self attention layer
                if name == 'self':
                    ref_feature = layer(ref_feature, ref_feature)
                    ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H))
            return ref_feature_list

        elif feat == "src":
            assert self.d_model == ref_feature[0].size(1)
            _, _, H, _ = ref_feature[0].shape

            ref_feature = [einops.rearrange(_, 'n c h w -> n (h w) c') for _ in ref_feature]

            src_feature = einops.rearrange(self.pos_encoding(src_feature), 'n c h w -> n (h w) c')

            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == 'self':
                    src_feature = layer(src_feature, src_feature)
                elif name == 'cross':
                    src_feature = layer(src_feature, ref_feature[i // 2])
                else:
                    raise KeyError
            return einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H)
        
        elif feat == "cross": 
            # assert self.d_model == ref_feature.size(1)
            _, _, H, _ = ref_feature.shape
            
            feature0 = ref_feature #* (nC2, C, H, W)
            feature1 = src_feature #* (nC2, C, H, W)
            
            feature0 = einops.rearrange(self.pos_encoding(feature0), 'n c h w -> n (h w) c')
            feature1 = einops.rearrange(self.pos_encoding(feature1), 'n c h w -> n (h w) c')
            
            pair_feat1 = torch.cat([feature0, feature1], dim=0)
            pair_feat2 = torch.cat([feature1, feature0], dim=0)
            
            #! for feature0
            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == 'self':
                    pair_feat1 = layer(pair_feat1, pair_feat1)
                elif name == 'cross':
                    pair_feat1 = layer(pair_feat1, pair_feat2)
                else:
                    raise KeyError
        
                
            return einops.rearrange(pair_feat1, 'n (h w) c -> n c h w', h=H), einops.rearrange(pair_feat1, 'n (h w) c -> n c h w', h=H)
            
        else:
            raise ValueError("Wrong feature name")



class FMT_with_pathway(nn.Module):
    def __init__(self,
            base_channels=8,
            FMT_config={
                'd_model': 32,
                'nhead': 8,
                'layer_names': ['self', 'cross'] * 4}):

        super(FMT_with_pathway, self).__init__()

        self.FMT = FMT(FMT_config)

        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 1, base_channels * 1, 3, padding=1, bias=False)
        
        # channel from 16 to 32
        # self.pre_conv16to32 = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=False)
        # self.pre_conv8to32 = nn.Conv2d(base_channels * 1, base_channels * 4, 1, bias=False)

    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y


    def forward(self, features, ref_idx=0):
        """forward.

        :param features: multiple views and multiple stages features
        """

        for nview_idx, feature_multi_stages in enumerate(features):
            if nview_idx == ref_idx: # ref view
                ref_fea_t_list = self.FMT(feature_multi_stages["stage1"].clone(), feat="ref")
                feature_multi_stages["stage1"] = ref_fea_t_list[-1]
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(feature_multi_stages["stage1"]), feature_multi_stages["stage2"]))
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(feature_multi_stages["stage2"]), feature_multi_stages["stage3"]))

            else: # src view
                feature_multi_stages["stage1"] = self.FMT([_.clone() for _ in ref_fea_t_list], feature_multi_stages["stage1"].clone(), feat="src")
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(feature_multi_stages["stage1"]), feature_multi_stages["stage2"]))
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(feature_multi_stages["stage2"]), feature_multi_stages["stage3"]))

        return features
    
    
    def extract_pair_feature(self, features, stages=["stage1"]):
        n_views = len(features)
        c, h, w = features[0]['stage1'].shape[1:]
        index_lists = [(a, b) for a in range(n_views - 1) for b in range(a + 1, n_views)] #* 4 views: 0-1, 0-2, 0-3, 1-2, 1-3, 2-3

        feature0_stages_list, feature1_stages_list = [], []   
        for stage in stages:
            cur_feat0, cur_feat1 = [], []
            for i_idx, j_idx in index_lists: #* image pair iteration
                cur_feat0.append(features[i_idx][stage])
                cur_feat1.append(features[j_idx][stage])
            cur_feat0 = torch.stack(cur_feat0, dim=1)
            cur_feat1 = torch.stack(cur_feat1, dim=1)
            cur_feat0 = cur_feat0.reshape(-1, *cur_feat0.shape[-3:])
            cur_feat1 = cur_feat1.reshape(-1, *cur_feat1.shape[-3:])
            feature0_stages_list.append(cur_feat0)
            feature1_stages_list.append(cur_feat1)
            
        ''' output:
            feature0_stages_list: 모든 stage에 대한 Left features [(nC2, C, H, W), ...]
            feature1_stages_list: 모든 stage에 대한 Right features [(nC2, C, H, W), ...]
        '''
        return feature0_stages_list, feature1_stages_list
    
    def extract_cross_features(self, features, ref_idx=0):
        
        ''' imgs: range [0, 1] 
            features: list of features from multiple views and multiple stages
        '''
        results_dict = {}
        aug_feat0_list = []
        aug_feat1_list = []
        
        features0_list, features1_list = self.extract_pair_feature(features) #* 오직 stage 1만 사용
        
        all_stages = len(features0_list) #* stage개수 (여기서는 1만 사용)
        n_views = len(features) #* view 개수 (4개)
        batch_size, C, H, W = features[0]['stage1'].shape
        
        #! cross attention
        for att_idx, stage_idx in enumerate(range(all_stages)): #* scale =1
            # load backbone features
            feature0, feature1 = features0_list[stage_idx], features1_list[stage_idx]
            """
            feature0과 feature1에는 각각 해당하는 stage의 모든 Pair로부터의 feature가 nC2 개만큼 (6개 If 4) 들어있음.
            """
            
            feature0, feature1 = self.FMT(feature0, feature1, feat="cross")
            
            
            aug_feat0_list.append(feature0.reshape(batch_size, int(feature0.shape[0] / batch_size), *feature0.shape[-3:]))
            aug_feat1_list.append(feature1.reshape(batch_size, int(feature1.shape[0] / batch_size), *feature1.shape[-3:]))

        results_dict.update({
            'aug_feat0s': aug_feat0_list,
            'aug_feat1s': aug_feat1_list,})

        return results_dict

    