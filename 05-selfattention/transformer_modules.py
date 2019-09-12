from torch import nn
import torch
import torch
import math
import numpy as np

# multi-head self attention
class MultiheadAttention(nn.Module):
    def __init__(self,
                 input_size,
                 key_size,
                 value_size,
                 output_size,
                 attention_dropout=0.1,
                 num_heads=8):
        super(MultiheadAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads

        # transformation for input query, key and value
        self.input_query_transform = nn.Linear(input_size, key_size)
        self.input_key_transform = nn.Linear(input_size, key_size)
        self.input_value_transform = nn.Linear(input_size, value_size)

        self.attention_softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_transform = nn.Linear(value_size, output_size)

    def split_heads(self, x, num_heads):
        batch, length, input_size = x.size()
        assert input_size % num_heads == 0, (
               "the input size should be a multiple of number of heads")
        new_dim = input_size // num_heads
        ans = x.view(batch, length, num_heads, new_dim).transpose(1, 2)
        return ans

    def combine_heads(self, x, num_heads):
        batch, _, length, new_dim = x.size()
        ans = x.transpose(1, 2).contiguous().view(batch, length, num_heads * new_dim)
        return ans

    def forward(self,
                query,
                bias=None):
        """
            query: query, key and value of self-attention, [batch_size, length, input_size]
            num_heads: number of heads
            bias: the bias to mask the padded words, [batch_size, length, length]
        """

        batch_size, length, _ = query.size()

        q = self.input_query_transform(query)
        k = self.input_key_transform(query)
        v = self.input_value_transform(query)

        q = self.split_heads(q, self.num_heads)
        k = self.split_heads(k, self.num_heads)
        v = self.split_heads(v, self.num_heads)

        key_size_per_head = self.key_size // self.num_heads

        # refer to the paper "Attention is all you need"
        q = q / math.sqrt(key_size_per_head)

        logits = torch.matmul(q, k.transpose(2, 3))

        # mask the padded words
        if bias is not None:
            bias = bias.unsqueeze(1).expand_as(logits)
            logits += bias

        # calculate the attention for each head
        attn = self.attention_softmax(logits)
        drop_attn = self.attention_dropout(attn)
        x = torch.matmul(drop_attn, v)

        # get attention score all heads
        attn = attn.view(batch_size, self.num_heads, length, length)

        # combine the attention heads
        x = self.combine_heads(x, self.num_heads)

        ans = self.output_transform(x)

        return ans, attn


class FeadForwadLayer(nn.Module):
    def __init__(self,
                 input_size,
                 filter_size,
                 output_size,
                 relu_dropout=0.0):
        super(FeadForwadLayer, self).__init__()
        self.mid_layer = nn.Linear(input_size, filter_size)
        self.out_layer = nn.Linear(filter_size, output_size)
        self.relu = nn.ReLU()
        self.relu_dropout = nn.Dropout(relu_dropout)

    def forward(self, x):
        t = self.relu(self.mid_layer(x))
        o = self.out_layer(self.relu_dropout(t))
        return o


class LayerNorm(nn.Module):
    def __init__(self,input_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(input_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)

    def forward(self, x):
        # get mean and std of x
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        norm_x = (x - mean) / (std + self.eps)
        return norm_x * self.scale + self.bias


class EncoderLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 filter_size,
                 dropout,
                 relu_dropout,
                 attention_dropout,
                 num_heads=8):
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.attention_dropout = attention_dropout

        self.ma = MultiheadAttention(input_size=hidden_size,
                                     key_size=hidden_size,
                                     value_size=hidden_size,
                                     output_size=hidden_size,
                                     attention_dropout=attention_dropout,
                                     num_heads=num_heads)
        self.ffn = FeadForwadLayer(input_size=hidden_size,
                                   filter_size=filter_size,
                                   output_size=hidden_size,
                                   relu_dropout=self.relu_dropout)
        self.ma_prenorm = LayerNorm(hidden_size)
        self.ffn_prenorm = LayerNorm(hidden_size)
        self.ma_postdropout = nn.Dropout(dropout)
        self.ffn_postdropout = nn.Dropout(dropout)

    def forward(self, x, bias=None):
        # layer normalization + multi-attention head
        y, _ = self.ma(self.ma_prenorm(x))
        # dropout + residual connection
        x = self.ma_postdropout(y) + x
        # layer normalization + feed forward layer
        y = self.ffn(self.ffn_prenorm(x))
        # dropout + residual connection
        ans = self.ffn_postdropout(y) + x
        return ans


def test_ma(query=None):
    batch_size = 50
    length = 20
    input_size = 512

    key_size = 1024
    value_size = 1024
    output_size = 512

    attention_dropout = 0.1
    num_heads = 8

    if query is None:
        query = torch.rand(batch_size, length, input_size)
    else:
        batch_size, length, input_size = query.shape

    multihead_attn = MultiheadAttention(input_size=input_size,
                                        key_size=key_size,
                                        value_size=value_size,
                                        output_size=output_size,
                                        attention_dropout=attention_dropout,
                                        num_heads=num_heads)

    # masking for padded words is mandatory for multi-head attention but it's not implemented in the demo
    ans, attn = multihead_attn(query)

    assert ans.shape == torch.Size([batch_size, length, output_size])
    assert attn.shape == torch.Size([batch_size, num_heads, length, length])
    return ans, attn


def test_el(x):
    batch_size, length, hidden_size = x.shape
    filter_size = 2048
    dropout = 0.1
    relu_dropout = 0.1
    attention_dropout = 0.1
    num_heads = 10

    encoder_layer = EncoderLayer(hidden_size, filter_size, dropout, relu_dropout, attention_dropout, num_heads)
    ans = encoder_layer(x)

    assert ans.shape == torch.Size([batch_size, length, hidden_size])
    return ans


def load_fasttext():
    import os
    ex_sents = "I went to a store to buy a candy \."
    words = ex_sents.split()
    for word in words:
        os.system("grep '^{} ' wiki-news-300d-1M.vec >> sample.vec".format(word))


def read_sample_vec():
    lines = open("sample.vec", "r").readlines()
    vec = [[float(t) for t in line.split()[1:]] for line in lines]
    return vec


if __name__ == '__main__':
    # load_fasttext()
    x = torch.tensor(read_sample_vec()).unsqueeze(0)

    # test a multi-head attention
    # attn: [batch_size, num_heads, length, length]
    ans_ma, attn = test_ma(x)
    first_attn = attn[:, 0, :, :].contiguous()
    print("First attention head")
    print(first_attn)
    second_attn = attn[:, 1, :, :].contiguous()
    print("Second attention head")
    print(second_attn)

    # test one encoder layer
    ans_el = test_el(x)
