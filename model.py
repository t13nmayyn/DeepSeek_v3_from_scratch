import  torch 
from torch import nn
import torch.nn.functional as F 
import math
from typing import Optional, Tuple, List
import math
from rope import RotaryPositionalEmbedding,apply_rotary

## RMS 
class RMSNorm(nn.Module):
    def __init__(self,eps:float):
        super().__init__()
        self.eps=eps
    def forward(self,x:torch.Tensor):
        rms=torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        return x/rms



# latent self attention 
class LatentAttention(nn.Module):
    def __init__(self,embed_dim,kv_latent_dim):
        super().__init__()
        self.embed_dim=embed_dim
        self.kv_latent_dim=kv_latent_dim
        self.rms_norm=RMSNorm(eps=1e-6)

        self.w_dkv=nn.Linear(self.embed_dim,self.kv_latent_dim,bias=False)
        self.wq=nn.Linear(self.embed_dim,self.embed_dim,bias=False)
        self.w_uv=nn.Linear(self.embed_dim,self.kv_latent_dim,bias=False)
        self.w_uk=nn.Linear(self.embed_dim,self.kv_latent_dim,bias=False) 
        # path2 
        self.rope_k=RotaryPositionalEmbedding(self.kv_latent_dim)
        self.rope_q=RotaryPositionalEmbedding(self.embed_dim)
        self.w_k_rope=nn.Linear(self.embed_dim,self.kv_latent_dim,bias=False)  # (embed_dim x latent_dim)
        self.w_q_rope=nn.Linear(self.kv_latent_dim,self.embed_dim)

        self.w_o=nn.Linear(self.embed_dim,self.embed_dim,bias=False)
        self.register_buffer('absorbed_k',None)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    def forward(self,x,kv_cache=None,kr_cache=None,past_length=0):
        batch_size,seq_len,embed_dim=x.shape
        if self.absorbed_k is None:
            absorbed=torch.matmul(self.wq.weight,self.w_uk.weight.transpose(-1,-2))
            self.register_buffer('absorbed_k',absorbed)

        # Pat : 2 
        k=self.w_k_rope(x)
        batch_size,seq_len,latent_dim=k.shape
        # chageing the k to one head at a time 
        k=k.view(batch_size,1,seq_len,latent_dim) # batch x n head x seqlen x latent_dim
        sin_k,cos_k=self.rope_k(seq_len,x.device)
        k=apply_rotary(k,sin_k,cos_k)
        n_kr=k.view(batch_size,seq_len,latent_dim)
        # calculating the c_kv
        n_c_kv=self.w_dkv(x)

        # for path :2 
        q=self.w_q_rope(n_c_kv)
        q=q.view(batch_size,1,seq_len,embed_dim)
        sin_q,cos_q=self.rope_q(seq_len,x.device)
        q_r=apply_rotary(q,sin_q,cos_q)
        q_r=q_r.view(batch_size,seq_len,embed_dim)
        if kv_cache is None:
            c_kv=n_c_kv
            k_r=n_kr
        else:                                            
            c_kv=torch.cat([kv_cache,n_c_kv],dim=1)
            k_r=torch.cat([kr_cache,n_kr],dim=1)

        S=c_kv.size(1)  # changed c_kv seq_len 
        atten_score_1=torch.matmul(x,torch.matmul(self.absorbed_k,c_kv.transpose(-1,-2)))
        atten_score_2=torch.matmul(q_r,k_r.transpose(-1,-2))
        atten_score=atten_score_1+atten_score_2
        atten_scores=atten_score/((self.embed_dim+self.embed_dim)**0.5)                            


        mask=torch.ones(seq_len,S,device=x.device)
        mask=torch.tril(mask,diagonal=past_length)
        attent_scores=atten_scores.masked_fill(~mask.bool(),float('-inf'))

        value=torch.matmul(c_kv,self.w_uv.weight)
        atten_weights=torch.softmax(attent_scores,dim=-1)
        out=torch.matmul(atten_weights,value)
        out=self.w_o(out)
         
        return out,c_kv,k_r


# a=LatentAttention(8,4)
# print(a(torch.randn(2,4,8)))



class MultiHeadLatentAttention(nn.Module):
    def __init__(self,n_heads,embd_dim,kv_latent_dim):
        super().__init__()


        assert embd_dim%n_heads ==0,"Embedding dim should be divisible to num heads"
        self.embd_dim=embd_dim
        self.n_heads=n_heads
        self.head_dim=self.embd_dim//self.n_heads
        self.kv_latent_dim=kv_latent_dim

        self.heads=nn.ModuleList([LatentAttention(self.head_dim,self.kv_latent_dim) for _ in range(self.n_heads)])

        self.w_o=nn.Linear(self.embd_dim,self.embd_dim)

    def forward(self,x,kv_cache=None,kr_cache=None,past_length=0):
        batch_size,seq_len,embed_dim=x.shape
        # now change batch_size x seq_len x embed_dim to batch_size x n_head x seq_len x head_dim
        x_=x.view(batch_size,seq_len,self.n_heads,self.head_dim)
        x_split=x_.transpose(1,2)
        head_out=[]
        head_ckv=[]
        head_kr=[]
        for i,head in enumerate(self.heads):
            x_i=x_split[:,i,:,:]  # spliting the x 

            if kv_cache is None:
                c_kv_i=None
                kr_i=None
            else:
                c_kv_i=kv_cache[:,i,:,:]   # spliting the c_kv according to the head
                kr_i=kr_cache[:,i,:,:]

            out,c_kv,k_r=head(x_i,kv_cache=c_kv_i,kr_cache=kr_i)
            head_out.append(out)
            head_ckv.append(c_kv)
            head_kr.append(k_r)
        concat_out=torch.cat(head_out,dim=-1)
        concat_ckv=torch.stack(head_ckv,dim=1)
        concat_kr=torch.stack(head_kr,dim=1)
        return concat_out,concat_ckv,concat_kr
   
attn = MultiHeadLatentAttention(n_heads=2, embd_dim=8, kv_latent_dim=4)

x1 = torch.randn(1, 4, 8)  # first token
out1, kv1,kr1 = attn(x1)  # kv1 shape: (1, 2, 1, 4)

# x2 = torch.randn(1, 1, 8)  # second token
# out2, kv2 = attn(x2, kv_cache=kv1)  # should return kv2 of shape (1, 2, 2, 4)

print(kv1.shape)  # should be (1, 2, 1, 4)
print(kr1.shape)          


# mixture of experts code

class NoisyTopKRouter(nn.Module):
    def __init__(self,n_experts,top_k,embed_dim):
        super().__init__()
        self.top_k=top_k
        self.n_experts=n_experts
        self.router=nn.Linear(embed_dim,self.n_experts,bias=False)
        self.noise_linear=nn.Linear(embed_dim,self.n_experts,bias=False)
        # bias 
        self.bias=nn.Parameter(torch.zeros(self.n_experts))
        # u = for updating the bias term  
        self.bias_update_speed=0.001 

        # normalize the weights 
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02 / math.sqrt(embed_dim))
        nn.init.normal_(self.noise_linear.weight, mean=0.0, std=0.02 / math.sqrt(embed_dim))
    def forward(self,x):
        batch_size,seq_len,embed_dim=x.shape
        x_flat=x.view(-1,x.size(-1))
        logits=self.router(x)
        noisy_logits=self.noise_linear(x)
        noise=torch.rand_like(logits)*F.softplus(noisy_logits)
        noise_logits=noise+logits+self.bias   # adding the bias terms for the load balancing 
        zeros=torch.full_like(noise_logits,fill_value=float('-inf'))
        topk_logits,indices=noise_logits.topk(k=self.top_k,dim=-1)
        sparse_logits=zeros.scatter_(dim=-1,index=indices,src=topk_logits)
        probs=F.softmax(sparse_logits,dim=-1)
        expert_selection_mask=F.one_hot(indices,self.n_experts).sum(dim=(0,1,2)).float()
        current_expert_load=expert_selection_mask.sum(0)
        total_tokens=x_flat.size(0)*self.top_k
        avg_token_load_per_expert=total_tokens/self.n_experts
        load_voilation=avg_token_load_per_expert-current_expert_load
        self.bias.data=self.bias.data+self.bias_update_speed*torch.sign(load_voilation)
        probs=probs.view(batch_size,seq_len,self.n_experts)
        indices=indices.view(batch_size,seq_len,self.top_k)
        return probs,indices


# expert
class Expert(nn.Module):
    def __init__(self,embed_dim:int,dropout:float):
        super().__init__()
        
        self.layer=nn.Sequential(
            nn.Linear(embed_dim,4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim,embed_dim),
            nn.Dropout(p=dropout)
        )
    def forward(self,x):
        return self.layer(x)


#sparsemoe

class SparseMoE(nn.Module):
    def __init__(self,embed_dim:int,dropout:float,n_experts:int,top_k:int):
        super().__init__()
        self.router=NoisyTopKRouter(n_experts=n_experts,top_k=top_k,embed_dim=embed_dim)
        self.experts=nn.ModuleList([Expert(embed_dim=embed_dim,dropout=dropout)  for _ in range(n_experts)])
    def forward(self,x):
        batch_size,seq_len,embed_dim=x.shape
        probs,indices=self.router(x)
        x_flat=x.view(-1,x.size(-1))
        probs_flat=probs.view(-1,probs.size(-1))
        indices_flat=indices.view(-1,indices.size(-1))
        out=torch.zeros_like(x_flat)
        for expert_idx,expert in enumerate(self.experts):
            mask=(indices_flat==expert_idx)
            selected=mask.any(-1)
            if selected.sum()==0:
                continue
            probs_selected=probs_flat[selected]
            selected_probs_for_expert=probs_selected[:,expert_idx]
            x_probs=selected_probs_for_expert.view(-1,1)
            x_split=x_flat[selected]
            expert_out=expert(x_split)
            out[selected]=out[selected]+expert_out*x_probs
        out=out.view(batch_size,seq_len,embed_dim)
        return out





smoe=SparseMoE(8,0.1,3,2)
print(torch.randn(2,4,8))

    

class Transformer(nn.Module):
    """
    A single block of the Transformer, combining multi-head attention and a sparse mixture of experts.
    """
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.embed_dim = config["embed_dim"]
        self.kv_latent_dim = config["kv_latent_dim"]
        self.n_experts = config["n_experts"]
        self.top_k = config["top_k"]
        self.dropout = config.get("dropout", 0.1)

        # Self-Attention Block
        self.attention = MultiHeadLatentAttention(
            n_heads=self.n_heads,
            embd_dim=self.embed_dim,
            kv_latent_dim=self.kv_latent_dim
        )
        self.norm1 = RMSNorm(eps=1e-6)

        # Feed-Forward Block (Mixture of Experts)
        self.moe = SparseMoE(
            embed_dim=self.embed_dim,
            dropout=self.dropout,
            n_experts=self.n_experts,
            top_k=self.top_k
        )
        self.norm2 = RMSNorm(eps=1e-6)

    def forward(self, x, kv_cache=None, kr_cache=None, past_length=0):
        # Attention part
        attn_out, new_kv_cache, new_kr_cache = self.attention(
            self.norm1(x),
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            past_length=past_length
        )
        x = x + attn_out

        # MoE part
        moe_out = self.moe(self.norm2(x))
        x = x + moe_out

        return x, new_kv_cache, new_kr_cache


model_config = {
    "embed_dim": 512,       # Embedding dimension (hidden size)
    "n_heads": 8,           # Number of attention heads (embed_dim must be divisible by this)
    "kv_latent_dim": 64,    # Per-head dim (usually embed_dim / n_heads)
    "ffn_dim": 2048,        # Feedforward hidden layer size (usually 4 * embed_dim)
    "num_layers": 6,        # Number of transformer blocks
    "dropout": 0.1,         # Dropout rate
    'vocab_size':50526,

    # MoE-specific parameters
    "n_experts": 4,         # Number of MoE experts
    "top_k": 2              # Top-k experts to route to
}


#creating the instant of the transformer block
transformer=Transformer(model_config)
print(torch.randn(1,4,8))