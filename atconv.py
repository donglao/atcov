import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class AttentionConv2D(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, attention_size=None, dropout=0.1):
        super(AttentionConv2D, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.attention_size = input_size if attention_size is None else attention_size
        self.dropout = torch.nn.Dropout(p=dropout)
        self.to_q = torch.nn.Linear(input_size, self.attention_size)
        self.to_k = torch.nn.Linear(input_size, self.attention_size)
        self.to_v = torch.nn.Linear(input_size, output_size)
        self.zero_padding = torch.nn.ZeroPad2d(int((kernel_size - 1) / 2))
        self.eps = 1e-8
        self.scale = self.attention_size ** -0.5

    def create_positional_encoding(self):
        d_model = self.input_size
        seq_length = self.kernel_size**2
        pos = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        
        return pe[:seq_length].permute(2,1,0).to(device)

    def expand_shift(self, K):
        _, _, h, w = K.shape
        K_expanded = self.zero_padding(K)
        # 'K_expanded' has shape [batch, attention_size, h+padding*2, w+padding*2, neighbor_size]

        K = torch.zeros_like(K)[:,:,:,:,None].repeat(1,1,1,1,self.kernel_size**2)

        pos = 0
        max_shift = int((self.kernel_size - 1) / 2)
        for i in range(-max_shift, max_shift + 1):
            for j in range(-max_shift, max_shift + 1):
                K[:,:,:,:,pos] = K_expanded[:,:,i+max_shift:i+h+max_shift,j+max_shift:j+w+max_shift]

        # 'K' has shape [batch, channel_size, h, w, neighbor_size]
        return K


    def compute_attention(self, K, Q):
        # 'Q' has shape [batch, attention_size, h, w]
        # 'K' has shape [batch, attention_size, h, w, neighbor_size]
        attention = torch.einsum('bahw,bahwp->bphw', Q, K) * self.scale
        attention = attention.softmax(dim=1) + self.eps
        # 'attention' has shape [batch, neighbor_size, h, w]
        return attention

    def forward(self, x):
        # 'x' has shape [batch, input_size, h, w]
        pos_emb = self.create_positional_encoding()
        # 'pos_emb' has shape [input_size, 1, neighbor_size]

        x_expanded = self.expand_shift(x)
        # 'x_expanded' has shape [batch, input_size, h, w, neighbor_size]
        x_expanded = x_expanded + pos_emb[None,:,:,None,:, ].expand_as(x_expanded)
        x_expanded = x_expanded.permute(0,4,2,3,1)
        # 'x_expanded' has shape [batch, neighbor_size, h, w, input_size]

        K = self.to_k(x_expanded)
        # 'K' has shape [batch, neighbor_size, h, w, attention_size] 
        x = x.permute(0,2,3,1)      
        Q = self.to_q(x).permute(0,3,1,2)
        V = self.to_v(x).permute(0,3,1,2)

        V_expanded = self.expand_shift(V)
        attention = self.compute_attention(K.permute(0,4,2,3,1), Q)
        
        # 'attention' has shape [batch, neighbor_size, h, w]
        output = torch.einsum('bchwn,bnhw->bchw', V_expanded, attention)

        return output
