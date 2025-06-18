from torch import nn 
import torch
import torch.nn.functional as F 




class RMSNorm(nn.Module):
    def __init__(self,embed_dim,eps:float=1e-8):
        super().__init__()
        self.eps=eps
    def forward(self,x):
        rms=torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)  + self.eps)
        return x/rms

rmsnorm=RMSNorm(8)
print(rmsnorm(torch.randn(1,4,8)))
print(torch.randn(1,4,8))
print(torch.randn(1,4,8)[:,0])

class MTP(nn.Module):
    def __init__(self,embed_dim,vocab_size,n_heads,  ):
        super().__init__()
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.n_heads=n_heads
        self.rmsnorm=RMSNorm(self.embed_dim)
        self.embeds=nn.Embedding(self.vocab_size,self.embed_dim)
        self.umembed=nn.Linear(self.embed_dim,self.vocab_size,bias=False)
        self.unembed_weight=self.umembed.weight
        self.projections=nn.ModuleList([nn.Linear(2*self.embed_dim,embed_dim)  for _ in range(self.n_heads)])
        self.transformers=nn.ModuleList([nn.TransformerDecoderLayer(self.embed_dim,self.n_heads) for _ in range(self.n_heads)])
    def forward(self,x,init_hidden:torch.Tensor=None):
        embeds=self.embeds(x)
        batch_size,seq_len,embed_dim=embeds.shape
        if init_hidden is None:
            h0_seq=embeds
        else:
            h0_seq=init_hidden
        outputs=[]   # batch_size X seq_len X vocab_size 
        
        max_i=seq_len-self.n_heads-1

        for i in range(0,max_i+1):
            h_prev=h0_seq[:,i,:]   # (batch_size x embedd)   # selecting the sequence 

            logits_k=[]

            # for multi tokens

            for k in range(self.n_heads):
                future_pos=i+(k+1)
                tok_embed=embeds[:,future_pos,:]     

                h_norm=self.rmsnorm(h_prev)   # this is normalizing the previous token 
                e_norm=self.rmsnorm(tok_embed) # normalzing the token embedding for the prev toekn 


                # cocatening all the token 
                merged=torch.cat([h_norm,e_norm],dim=-1)

                proj=self.projections(merged[k])  # takin the projection for the kth idx

                # transformers block 
                x=proj.unsqueeze(0)  # we provided a extra dim 

                x=self.transformers[k](x)  # out (1,seq_len , embed_dim )

                # we have to remove the extra dim as we will be appending each out 
                x_curr=x.squeeze(0)

                logits=self.umembed(x_curr)
                logits_k.append(logits)

                # now passing the predicted token for the prediction of second token 
                h_prev=x_curr
            logits_k=torch.stack(logits_k,dim=1)
            outputs.append(logits_k)
        out=torch.stack(outputs,dim=0)
        out=out.permute(1,0,2,3).contiguous()
        return out


embed_dim = 32
vocab_size = 100
n_heads = 4
batch_size = 2
seq_len = 10

# Instantiate the model
mtp = MTP(embed_dim=embed_dim, vocab_size=vocab_size, n_heads=n_heads)

# FIX 1: Create a tensor of integer indices (type long) for the input
# The values must be within the vocabulary size (0 to 99)
input_indices = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

# Run the model
try:
    output_logits = mtp(input_indices)
    print("Model ran successfully!")
    print("Output shape:", output_logits.shape)
    # Expected output shape: (batch_size, seq_len - n_heads, n_heads, vocab_size)
    # (2, 6, 4, 100)
except Exception as e:
    print(f"An error occurred: {e}")















