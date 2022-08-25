import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class AttentionConv2D(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, attention_size=None, dropout=0.1, num_heads=2):
        super(AttentionConv2D, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.attention_size = round(input_size/num_heads) if attention_size is None else attention_size
        self.output_size_single_head = round(output_size/num_heads)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.to_q = torch.nn.Linear(input_size, self.attention_size * num_heads)
        self.to_k = torch.nn.Linear(input_size, self.attention_size * num_heads)
        self.to_v = torch.nn.Linear(input_size, self.output_size_single_head * num_heads)
        self.to_pos = torch.nn.Linear(self.input_size, self.attention_size * num_heads)
        self.final = torch.nn.Linear(self.output_size_single_head * num_heads, self.output_size)
        self.layernorm = torch.nn.LayerNorm(self.input_size)
        self.num_heads = num_heads

        self.ReLU = torch.nn.ReLU()

        self.zero_padding = torch.nn.ZeroPad2d(int((kernel_size - 1) / 2))
        self.mirror_padding = torch.nn.ReflectionPad2d(int((kernel_size - 1) / 2))
        self.eps = 1e-8
        self.scale = self.attention_size ** -0.5

    def create_positional_encoding(self):
        d_model = self.input_size
        seq_length = self.kernel_size**2
        pos = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_length, d_model + d_model%2)

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe[:,0:d_model]

        return pe[:seq_length].to(device)

    def expand_shift(self, K):
        K = K.permute(0,3,1,2)
        _, _, h, w = K.shape
        K_expanded = self.zero_padding(K)
        # 'K_expanded' has shape [batch, attention_size, h+padding*2, w+padding*2, neighbor_size]

        K = torch.zeros_like(K)[:,:,:,:,None].repeat(1,1,1,1,self.kernel_size**2)

        pos = 0
        max_shift = int((self.kernel_size - 1) / 2)
        for i in range(-max_shift, max_shift + 1):
            for j in range(-max_shift, max_shift + 1):
                K[:,:,:,:,pos] = K_expanded[:,:,i+max_shift:i+h+max_shift,j+max_shift:j+w+max_shift]
                pos += 1

        # 'K' has shape [batch, channel_size, h, w, neighbor_size]
        return K


    def compute_attention(self, K, Q):
        # 'Q' has shape [batch, h, w, attention_size]
        # 'K' has shape [batch, attention_size, h, w, neighbor_size]

        attention = torch.einsum('bhwa,bahwp->bphw', Q, K) * self.scale
#        attention = torch.einsum('bahwp,bahwp->bphw', Q, K) * self.scale
        attention = attention.softmax(dim=1) + self.eps
        # 'attention' has shape [batch, neighbor_size, h, w]
        return attention

    def forward(self, x):
        # b, c, h, w = x.shape
        # 'x' has shape [batch, input_size, h, w]
        pos_emb = self.create_positional_encoding()
        pos_k = self.to_pos(pos_emb).permute(1,0)
        # 'pos_emb' has shape [neighbor_size, attention_size]

        x = x.permute(0,2,3,1)
        x = self.layernorm(x)
        K = self.to_k(x)

        K_no_pos = self.expand_shift(K)
        K = K_no_pos + pos_k[None, :, None, None, :] #- bias_k[None, :,None, None, None]

        Q = self.to_q(x)

        V = self.to_v(x)
        V = self.expand_shift(V)
        
        x = []
        for i in range(self.num_heads):
            attention = self.compute_attention(K[:,i*self.attention_size:(i+1)*self.attention_size,:,:,:], Q[:,:,:,i*self.attention_size:(i+1)*self.attention_size])
            # 'attention' has shape [batch, neighbor_size, h, w]
            x.append(torch.einsum('bchwn,bnhw->bhwc', V[:,i*self.output_size_single_head:(i+1)*self.output_size_single_head,:,:,:], attention))
        
        output = torch.cat(x,3)
        output = self.final(output)
        output = output.permute(0,3,1,2)

        return output[:,:,::self.stride, ::self.stride]
