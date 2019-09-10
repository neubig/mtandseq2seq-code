
import torch
from torch import nn
# very simple example code to calculate attention

class DotProdAttn(nn.Module):
  def __init__(self, d_model):
    super(DotProdAttn, self).__init__()
  
  def forward(self, q, k, v):
    batch_size, d_q = q.size()
    batch_size, len_k, d_k = k.size()
    batch_size, len_v, d_v = v.size()
    assert d_k == d_q 
    assert len_k == len_v
    
    # (batch_size, len_k, d_k)
    att_score_hidden = torch.bmm(q.unsqueeze(1), k.transpose(1, 2))
    # (batch_size, len_k)
    att_score_weights = torch.softmax(att_score_hidden, dim=-1)
    att_score = torch.softmax(att_score_weights, dim=-1)

    ctx = torch.bmm(att_score, v).squeeze(1)
    return ctx, att_score



class MlpAttn(nn.Module):
  def __init__(self, d_model):
    super(MlpAttn, self).__init__()
    self.w_trg = nn.Linear(d_model, d_model)
    self.w_att = nn.Linear(d_model, 1)
  
  def forward(self, q, k, v):
    batch_size, d_q = q.size()
    batch_size, len_k, d_k = k.size()
    batch_size, len_v, d_v = v.size()
    assert d_k == d_q 
    assert len_k == len_v
    
    # (batch_size, len_k, d_k)
    att_score_hidden = torch.tanh(k + self.w_trg(q).unsqueeze(1))
    # (batch_size, len_k)
    att_score_weights = self.w_att(att_score_hidden).squeeze(2)
    att_score = torch.softmax(att_score_weights, dim=-1)

    ctx = torch.bmm(att_score.unsqueeze(1), v).squeeze(1)
    return ctx, att_score

if __name__ == "__main__":
  mlp_attn = MlpAttn(2)
  dotprod_attn = DotProdAttn(2)

  src_encs = torch.FloatTensor([[[-1, 2], [2, 4], [3, 5]]])
  trg_enc = torch.FloatTensor([[1, 2]])

  print("src encodings:")
  print(src_encs)

  print("target encoding:")
  print(trg_enc)


  print("mlp attention scores")
  ctx, attn_score = mlp_attn(trg_enc, src_encs, src_encs)
  print(attn_score)
  
  print("dot prod attention scores")
  ctx, attn_score = dotprod_attn(trg_enc, src_encs, src_encs)
  print(attn_score)
