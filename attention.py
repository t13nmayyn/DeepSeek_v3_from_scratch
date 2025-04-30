import torch 
from torch import nn



# Attention BLock 
class LatentSelfAttention(nn.Module): 
    def __init__(self,embd_dim,kv_latent_dim):
        """Args --- : 
        embed_dim (int) :Total Dimension of the token embeddings
        kv_latent_dim  (int) : Dimensions of the latent vectors used for the attention  
        batch_size : No of batches present in the input token 
        seq_len : Length of the input token_sequence
        kv_cache :  
        C_kv=X @ W_dkv 
        Q=X @ W_q 
        K=C_kv @ W_uK 
        V=C_kv @ W_uV
        S= changed seq_len after caching 

        so we are caching only C_kv this will reduced our memory usage while inference 
        attention_scr = Q @ K.T 
        (X @ W_Q) @ (C_KV @ W_uK).T
        so we are storing only W_Q and W_UK.T in register_buffer precomputed 
        
        
        
        """
        super().__init__()
        self.embd_dim=embd_dim
        self.kv_latent_dim=kv_latent_dim
        self.w_q=nn.Linear(self.embd_dim,self.embd_dim,bias=False)  # Wq(embd x embd) 
        self.w_dkv=nn.Linear(self.embd_dim,self.kv_latent_dim,bias=False) # w_dkv(embd x kv_latent_dim)
        self.w_uk=nn.Linear(self.embd_dim,self.kv_latent_dim,bias=False) 
        # Its weights shape is self.w_uk.weight is W_uk & W_uv (kv_latent_dim x embed_dim)  
        self.w_uv=nn.Linear(self.embd_dim,self.kv_latent_dim,bias=False)
        self.w_o=nn.Linear(self.embd_dim,self.embd_dim,bias=False)   # W_o ( embd x embd )
        self.register_buffer('absorbed_k',None) 
        """ Here in register_buffer we are storing W_q  and W_uk """
        self.norm=nn.LayerNorm(kv_latent_dim)
    def forward(self,x,kv_cache=None,past_length=0): 
        batch_size,seq_len,embd_dim=x.shape 
        if self.absorbed_k  is None: 
            absorbed=torch.matmul(self.w_q.weight,self.w_uk.weight.transpose(-1,-2))
            self.register_buffer('absorbed_k',absorbed)
        n_c_kv=self.norm(self.w_dkv(x))   
        if kv_cache is None: 
            c_kv=n_c_kv
        else:
            c_kv=torch.cat([kv_cache,n_c_kv],dim=1)
        S=x.size(1)# new seq_len if we cached 
        #(X @ W_Q) @ (C_KV @ W_uK).T
        atten_scores=torch.matmul(x,torch.matmul(self.absorbed_k,c_kv.transpose(-1,-2)))
        atten_scores=atten_scores/(self.kv_latent_dim**0.5)
        # masking the atten weight 
        mask=torch.ones(seq_len,S,device=x.device)
        mask=torch.tril(mask,diagonal=past_length)
        atten_scores=atten_scores.masked_fill(~mask.bool(),float('-inf'))
        # calc the value 
        value=torch.matmul(c_kv,self.w_uv.weight)  #   (Sx kv_latent) @ (kv_lanet x embd)-> 
        # (S x Embed)
        # calculating weights 
        atten_weights=torch.softmax(atten_scores,dim=-1)
        out=torch.matmul(atten_weights,value)  
        out=self.w_o(out)
        return out,c_kv



#Testing 
mla=LatentSelfAttention(8,4)
mla(torch.randn(1,4,8))


# Multi Head Latent Attention BLock 

class MultiHeadLatentAttention(nn.Module):
    def __init__(self,n_heads,embd_dim,kv_latent_dim):
        super().__init__()
        """Args --- : 
        embed_dim (int) :Total Dimension of the token embeddings
        kv_latent_dim  (int) : Dimensions of the latent vectors used for the attention  
        batch_size : No of batches present in the input token 
        seq_len : Length of the input token_sequence
        kv_cache :  Used during inference 
        n_head (int) : No of num_heads  
        head_dim (int) : Dimension of the heads 
        S= changed seq_len after caching 
        C_kv=X @ W_dkv 
        Q=X @ W_q 
        K=C_kv @ W_uK 
        V=C_kv @ W_uV

        so we are caching only C_kv this will reduced our memory usage while inference 
        attention_scr = Q @ K.T 
        (X @ W_Q) @ (C_KV @ W_uK).T
        so we are storing only W_Q and W_UK.T in register_buffer precomputed 
        
        
        
        """
        assert embd_dim%n_heads ==0,"Embedding dim should be divisible to num heads"
        self.embd_dim=embd_dim
        self.n_heads=n_heads
        self.head_dim=self.embd_dim//self.n_heads
        self.kv_latent_dim=kv_latent_dim

        self.heads=nn.ModuleList([LatentSelfAttention(self.head_dim,self.kv_latent_dim) for _ in range(self.n_heads)])

        self.w_o=nn.Linear(self.embd_dim,self.embd_dim)
    def forward(self,x,kv_cache=None,past_length=0): 
        batch_size,seq_len,embd_dime=x.shape   
        x_split=x.view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        # outputs of per head 
        head_out=[]
        head_ckv=[]
        for i,head in enumerate(self.heads): 
            x_i=x_split[:,i] # spliting query for head 
            if kv_cache is not None: 
                cache_i=kv_cache[:,i,:,:] 
            else: 
                cache_i=None 
            out,c_kv=head(x_i,kv_cache=cache_i)
            head_out.append(out)
            head_ckv.append(c_kv)
        concat_out=torch.cat(head_out,dim=-1)
        concat_ckv=torch.stack(head_ckv,dim=1)
        return concat_out,concat_ckv
 

# testing kv attention & kv cache 
attn = MultiHeadLatentAttention(n_heads=2, embd_dim=8, kv_latent_dim=4)

x1 = torch.randn(1, 4, 8)  # first token
out1, kv1 = attn(x1)  # kv1 shape: (1, 2, 1, 4)

x2 = torch.randn(1, 1, 8)  # second token
out2, kv2 = attn(x2, kv_cache=kv1)  # should return kv2 of shape (1, 2, 2, 4)

print(kv1.shape)  # should be (1, 2, 1, 4)
print(kv2.shape)