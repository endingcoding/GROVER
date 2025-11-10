import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from fasterkan import FasterKANLayer

class Encoder_overall(Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dim_in_feat_omics3, dim_out_feat_omics3, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_in_feat_omics3 = dim_in_feat_omics3
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dim_out_feat_omics3 = dim_out_feat_omics3
        self.dropout = dropout
        self.act = act
        
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        self.encoder_omics3 = Encoder(self.dim_in_feat_omics3, self.dim_out_feat_omics3)
        self.decoder_omics3 = Decoder(self.dim_out_feat_omics3, self.dim_in_feat_omics3)

        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_omics3 = AttentionLayer(self.dim_out_feat_omics3, self.dim_out_feat_omics3)  
        self.switch_moe = GROVERMoE(dim=self.dim_out_feat_omics1, num_experts=3)

        self.mlp = nn.Sequential(nn.Linear(self.dim_out_feat_omics1 + self.dim_out_feat_omics2 + self.dim_out_feat_omics3, dim_out_feat_omics1))                  
                
    def forward(self, features_omics1, features_omics2, features_omics3, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2, adj_spatial_omics3, adj_feature_omics3):

        # graph1
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
        emb_latent_spatial_omics3 = self.encoder_omics3(features_omics3, adj_spatial_omics3)
        
        # graph2
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)
        emb_latent_feature_omics3 = self.encoder_omics3(features_omics3, adj_feature_omics3)

        # within-modality attention aggregation layer
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1) #RNA
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2) #ADT
        emb_latent_omics3, alpha_omics3 = self.atten_omics3(emb_latent_spatial_omics3, emb_latent_feature_omics3) #IMG

        # MoE 模态选择
        gate_input = (emb_latent_omics1 + emb_latent_omics2 + emb_latent_omics3) / 3
        expert_inputs = [emb_latent_omics1, emb_latent_omics2, emb_latent_omics3]

        emb_latent_combined, top2_weight = self.switch_moe(gate_input, expert_inputs)
        #emb_latent_combined = (emb_latent_omics1 + emb_latent_omics2 + emb_latent_omics3)
        #emb_latent_combined = self.mlp(torch.cat([emb_latent_omics1, emb_latent_omics2, emb_latent_omics3], dim=-1)) # shape: (N, 3D)
        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)
        emb_recon_omics3 = self.decoder_omics3(emb_latent_combined, adj_spatial_omics3)
        
        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_omics3':emb_latent_omics3,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'emb_recon_omics3':emb_recon_omics3,
                   'alpha_omics1':alpha_omics1,
                   'alpha_omics2':alpha_omics2,
                   'alpha_omics3':alpha_omics3,
                   }
        
        return results
    
class Encoder(Module): 
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.kan = FasterKANLayer(input_dim=out_feat, output_dim=out_feat)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj, visualize=False):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        x = self.kan(x)   
        return x

    
class Decoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.kan = FasterKANLayer(input_dim=out_feat, output_dim=out_feat)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)                  
        x = self.kan(x)
        return x                  

class AttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha
    
    
from zeta.nn import FeedForward     
class GROVERGate(nn.Module):
    def __init__(self, dim, num_experts: int, capacity_factor: float = 1.0, epsilon: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        gate_scores = F.softmax(self.w_gate(x), dim=-1)  # (N, num_experts)
        topk_scores, topk_indices = gate_scores.topk(2, dim=-1)  # Top-2
        return gate_scores, topk_scores, topk_indices


class GROVERMoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 3, threshold: float = 0.3):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.threshold = threshold
        self.experts = nn.ModuleList([
            FeedForward(dim, dim, mult=4) for _ in range(num_experts)
        ])
        self.gate = GROVERGate(dim, num_experts)
        self.latest_stats = None

    def forward(self, x, expert_inputs):
        gate_scores, _, _ = self.gate(x)
        expert_outputs = [expert(inp) for expert, inp in zip(self.experts, expert_inputs)]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        mask = gate_scores >= self.threshold
        masked_scores = gate_scores * mask.float()
        selected_counts = mask.sum(dim=1)
        fallback_mask = (selected_counts == 0).unsqueeze(1)
        normed_scores = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + 1e-6)
        fused = torch.sum(normed_scores.unsqueeze(-1) * expert_outputs, dim=1)

        top1_indices = gate_scores.argmax(dim=1)
        fallback_output = expert_outputs[torch.arange(expert_outputs.size(0)), top1_indices]
        fused = fallback_mask * fallback_output + (~fallback_mask) * fused

        with torch.no_grad():
            mean_scores = gate_scores.mean(dim=0)
            selection_ratio = mask.float().mean(dim=0)
            top1_counts = torch.bincount(top1_indices, minlength=self.num_experts).float()
            top1_ratio = top1_counts / top1_indices.numel()

            self.latest_stats = {
                "mean_scores": mean_scores,
                "selection_ratio": selection_ratio,
                "top1_ratio": top1_ratio
            }

        return fused, gate_scores




        