import torch 
from torch import nn 
import torch.nn.functional as F 
import math

class Router(nn.Module):
    def __init__(self,n_expert,embd_dim,top_k):
        super().__init__()
        self.embd_dim=embd_dim
        self.n_expert=n_expert
        self.topk=top_k
        self.topK_router=nn.Linear(in_features=self.embd_dim,out_features=self.n_expert,bias=False)
    def forward(self,x):
        # x: batch_size x seq_len x embdim   
        logits=self.topK_router(x) ## out : (seq_len x n_expert)
        top_k_logits,indices=logits.topk(self.topk,dim=-1)
        zeros=torch.full_like(logits,fill_value=float('-inf'))

        top_k_logits=zeros.scatter_(dim=-1,index=indices,src=top_k_logits)
        probs=F.softmax(top_k_logits,dim=-1)
        return probs,indices
        
router=Router(4,8,2)
print(router(torch.randn(1,6,8)))

### Add some noise to router 
class NoisyTopKRouter(nn.Module):
    def __init__(self,top_k,embd_dim,n_expert):
        super().__init__()
        self.top_k=top_k
        self.embd_dim=embd_dim
        self.n_expert=n_expert
        self.router=nn.Linear(self.embd_dim,self.n_expert,bias=False)
        self.noise_linear=nn.Linear(self.embd_dim,self.n_expert,bias=False)
        # bais term for load balacing
        self.bias=nn.Parameter(torch.zeros(self.n_expert))
        self.bias_update_speed=0.001 

        # normalize the weights 
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02 / math.sqrt(embd_dim))
        nn.init.normal_(self.noise_linear.weight, mean=0.0, std=0.02 / math.sqrt(embd_dim))

    def forward(self,x):

        batch_size,seq_len,embed_dim=x.shape
        # convering 3D to 2D
        x_flat=x.view(-1,self.embd_dim)

        logits=self.router(x)
        noise_logits=self.noise_linear(x)
        # creating some noise for better distribution of token in the router 
        noise=torch.rand_like(logits)*F.softplus(noise_logits)
        # creating a noisy logit
        noisy_logits=noise+logits+self.bias
        zeros=torch.full_like(noisy_logits,fill_value=float('-inf'))
        topk_logits,indices=noisy_logits.topk(self.top_k,dim=-1)
        sparse_logits=zeros.scatter_(dim=-1,index=indices,src=topk_logits)
        probs=F.softmax(sparse_logits,dim=-1)

        # load balancing : Auxillary loss free load balancing 
        expert_selection_mask=F.one_hot(indices,self.n_expert).sum(1).float()
        current_expert_load=expert_selection_mask.sum(0)
        total_selection=x_flat.size(0)*self.top_k
        avg_token_load_per_expert=total_selection/self.n_expert
        load_voilation=avg_token_load_per_expert-current_expert_load
        self.bias.data=self.bias.data+self.bias_update_speed*torch.sign(load_voilation)
        probs=probs.view(batch_size,seq_len,self.n_expert)
        indices=indices.view(batch_size,seq_len,self.top_k)

        return probs,indices

router=NoisyTopKRouter(2,8,3)
print(router(torch.randn(1,6,8)))

class Expert(nn.Module):
  def __init__(self,embd_dim,dropout):
    super().__init__()
    self.embd_dim=embd_dim
    self.net=nn.Sequential(
      nn.Linear(self.embd_dim,4*self.embd_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(4*self.embd_dim,self.embd_dim),

    )

  def forward(self,x):
    return self.net(x)




# This block is without multiplying  the probs to the expert output which is mathematically wrong

class SparseMoE(nn.Module):
    def __init__(self,embd_dim,n_expert,dropout,top_k):
        super().__init__()
        self.embd_dim=embd_dim
        self.top_k=top_k
        self.n_expert=n_expert
        self.router=NoisyTopKRouter(self.top_k,self.embd_dim,self.n_expert)
        self.experts=nn.ModuleList([Expert(self.embd_dim,dropout) for _ in range(self.n_expert)])
    def forward(self,x):
      batch_size,seq_len,embd_dim=x.shape
      probs,indices=self.router(x)
      print('Probs : ',probs," \n probs.shape amd Indices",probs.shape,indices)
      #now converting the 3D Tensor into 2D Tensor
      x_flat=x.view(-1,x.size(-1))
      out=torch.zeros_like(x_flat)
      indices_flat=indices.view(-1,indices.size(-1))
      probs_flat=probs.view(-1,probs.size(-1))

      # expert loop 
      for idx,expert in enumerate(self.experts):
        # checking if the expert is selected from the indices 
        mask=(idx==indices_flat) 
        selected=mask.any(-1)   # checking from the last idx


        if selected.sum()==0:
          continue

        selected_input=x_flat[selected] #for eg x[1,2,3]=[0..,0..]
        expert_output=expert(selected_input)

        out[selected]=out[selected]+expert_output
      out=out.view(batch_size,seq_len,embd_dim)
      return out



moe=SparseMoE(8,4,0.1,2)
print(moe(torch.randn(2,4,8)))

    

class SparseMOElayer(nn.Module):
  def __init__(self,embed_dim,drop,top_K,n_expert):
    super().__init__()
    self.router=Router(n_expert,embed_dim,top_K)
    self.top_K=top_K
    self.experts=nn.ModuleList([Expert(embed_dim,drop) for _ in range(n_expert)])
    self.shared_expert=Expert(embd_dim=embed_dim,dropout=drop)
  def forward(self,x):
    batch_size,seq_len,embed_dim=x.shape
    probs,idx=self.router(x)

    x_flat=x.view(-1,x.size(-1))
    idx_flat=idx.view(-1,idx.size(-1))
    probs_flat=probs.view(-1,probs.size(-1))

    output=torch.zeros_like(x_flat)

    for i,expert in enumerate(self.experts):
      
      mask=(idx_flat==i)

      # taking the selected the experts
      selected=mask.any(-1) # selecting from the last dim

      if selected.sum()==0:
        continue

      x_selected=x_flat[selected]
      # for selectin the probs 

      x_probs=probs_flat[selected]
      x_probs=x_probs[:,i]     # taking the entire columns of the expert
      probs_selected=x_probs.view(-1,1) # keeping the last dim as 1 so the matrix can multiply the scalar 

      expert_out=expert(x_selected)
      #shared experts output 
      shared_expert_out=self.shared_expert(x_selected)
      output[selected]=output[selected]+probs_selected*expert_out + 0.1*shared_expert_out
    output=output.view(batch_size,seq_len,embed_dim)
    return output

smoe_layer=SparseMOElayer(8,0.1,2,3)
smoe_layer(torch.randn(2,4,8))