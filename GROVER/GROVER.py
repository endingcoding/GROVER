import torch
from tqdm import tqdm
import torch.nn.functional as F
from model import Encoder_overall
from preprocess import adjacent_matrix_preprocessing
import numpy as np

class Train_GROVER:
    def __init__(self, 
        data,
        datatype = 'Triplet',
        device= torch.device('cpu'),
        random_seed = 2025,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 1, 1, 2, 2, 2]
        ):
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adata_omics3 = self.data['adata_omics3']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, self.adata_omics3)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_spatial_omics3 = self.adj['adj_spatial_omics3'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        self.adj_feature_omics3 = self.adj['adj_feature_omics3'].to(self.device)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.features_omics3 = torch.FloatTensor(self.adata_omics3.obsm['feat'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        self.n_cell_omics3 = self.adata_omics3.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_input3 = self.features_omics3.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        self.dim_output3 = self.dim_output
    
    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2, self.dim_input3, self.dim_output3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            #results = self.model(self.features_omics1, self.features_omics2, self.features_omics3, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2, self.adj_spatial_omics3, self.adj_feature_omics3)
            results = self.model(
                self.features_omics1, self.features_omics2, self.features_omics3,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
                self.adj_spatial_omics3, self.adj_feature_omics3,  
            )

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            self.loss_recon_omics3 = F.mse_loss(self.features_omics3, results['emb_recon_omics3'])
            
            z1 = results['emb_latent_omics1']  # RNA
            z2 = results['emb_latent_omics2']  # ADT
            z3 = results['emb_latent_omics3']  # IMG

            # masked infoNCE loss
            z1_sim_mask = build_similarity_mask(z1)
            z2_sim_mask = build_similarity_mask(z2)
            z3_sim_mask = build_similarity_mask(z3)
            loss_contrast_rna_adt = (masked_info_nce_loss(z1, z2, z1_sim_mask) + masked_info_nce_loss(z2, z1, z2_sim_mask)) / 2
            loss_contrast_rna_img = (masked_info_nce_loss(z1, z3, z1_sim_mask) + masked_info_nce_loss(z3, z1, z3_sim_mask)) / 2            
            loss_contrast_adt_img = (masked_info_nce_loss(z2, z3, z2_sim_mask) + masked_info_nce_loss(z3, z2, z3_sim_mask)) / 2 
            
            
            loss = self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2 + self.weight_factors[2] * self.loss_recon_omics3 \
                 + self.weight_factors[3] * loss_contrast_rna_adt + self.weight_factors[4] * loss_contrast_rna_img + self.weight_factors[5] * loss_contrast_adt_img 
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step() 

        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.features_omics3, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2, self.adj_spatial_omics3, self.adj_feature_omics3)
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_omics3 = F.normalize(results['emb_latent_omics3'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_latent_omics3': emb_omics3.detach().cpu().numpy(),
                  'GROVER': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha_omics3': results['alpha_omics3'].detach().cpu().numpy(),
                }
        return output
    
import torch
import torch.nn.functional as F

def masked_info_nce_loss(z1, z2, mask=None, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()  # 数值稳定

    batch_size = z1.size(0)
    labels = torch.arange(batch_size).long().to(z1.device)

    if mask is not None:
        logits = logits.masked_fill(~mask, float('-inf'))

    loss = F.cross_entropy(logits, labels)
    return loss

from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_mask(z, threshold=0.8):
    z_cpu = z.detach().cpu().numpy()
    sim = cosine_similarity(z_cpu)
    mask = sim < threshold 
    np.fill_diagonal(mask, True)  
    return torch.tensor(mask, dtype=torch.bool, device=z.device)
    
      

    
        
    
    