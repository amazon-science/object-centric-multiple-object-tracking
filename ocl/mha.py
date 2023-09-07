import torch
import torch.nn as nn
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # if mask is not None:
        #     attn = attn.masked_fill(mask == 0, -1e9)
        if mask is not None:
            bias = (1-mask)*(-1e9)
            attn = attn * mask + bias


        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention_for_index(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        '''
        nn.init.constant_(self.w_qs.weight, 0)
        nn.init.constant_(self.w_ks.weight, 0)
        nn.init.constant_(self.w_vs.weight, 0)
        '''

        # nn.init.eye_(self.w_qs.weight)
        nn.init.eye_(self.w_ks.weight)
        nn.init.eye_(self.w_vs.weight)
        # nn.init.eye_(self.w_qs.weight)

        # self.w_vs.eval()
        # self.w_ks.eval()
        # self.w_qs.eval()
        # self.w_vs.weight.requires_grad = False
        # self.w_ks.weight.requires_grad = False
        # self.w_qs.weight.requires_grad = False

        # nn.init.eye_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)  #temperature=d_k ** 0.5

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # print("w_qs:", self.w_qs.weight)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)


        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # print (self.w_qs.weight.abs().sum(), self.w_ks.weight.abs().sum(), self.w_vs.weight.abs().sum())
        # print ('kqv', q.abs().sum(), k.abs().sum(), v.abs().sum())

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn2 = self.attention(q, k, v, mask=mask)


        # attn = attn1 + attn2
        # attn = attn1.unsqueeze(1)
        attn = attn2

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # q += residual

        q = self.layer_norm(q)

        attn = torch.mean(attn, 1)
        return q, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)



        nn.init.eye_(self.w_qs.weight)
        nn.init.eye_(self.w_ks.weight)
        nn.init.eye_(self.w_vs.weight)
        nn.init.eye_(self.fc.weight)
        self.attention = ScaledDotProductAttention(temperature=0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        #print("w_qs:", self.w_qs.weight)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        # q += residual
        # q = self.fc2(q)
        # do not norm
        # q = self.layer_norm(q)

        attn = torch.mean(attn, 1)
        return q, attn



class MultiHeadAttention_dotversion_merge(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.attention = ScaledDotProductAttention(temperature=0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        #print("w_qs:", self.w_qs.weight)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))


        attn = torch.mean(attn, 1)
        return q, attn

class MultiHeadAttention_dotversion_index(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0):
        super().__init__()

        self.n_head = n_head

        '''
        nn.init.constant_(self.w_qs.weight, 0)
        nn.init.constant_(self.w_ks.weight, 0)
        nn.init.constant_(self.w_vs.weight, 0)
        '''
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)  #temperature=d_k ** 0.5

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # print("w_qs:", self.w_qs.weight)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)


        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)
        # print (self.w_qs.weight.abs().sum(), self.w_ks.weight.abs().sum(), self.w_vs.weight.abs().sum())
        # print ('kqv', q.abs().sum(), k.abs().sum(), v.abs().sum())

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn2 = self.attention(q, k, v, mask=mask)


        # attn = attn1 + attn2
        # attn = attn1.unsqueeze(1)
        attn = attn2

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(q)
        # q += residual

        q = self.layer_norm(q)

        attn = torch.mean(attn, 1)
        return q, attn