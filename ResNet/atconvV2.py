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
        self.to_q = torch.nn.Linear(input_size, self.attention_size * 2)
        self.to_k = torch.nn.Linear(input_size, self.attention_size * 2)
        self.to_v = torch.nn.Linear(input_size, self.attention_size * 2)
        self.to_pos = torch.nn.Linear(self.input_size, self.attention_size)
        self.to_posk = torch.nn.Linear(self.attention_size, self.attention_size)
        self.to_posv = torch.nn.Linear(self.attention_size, self.output_size)

        self.to_q2 = torch.nn.Linear(self.attention_size * 2, self.attention_size)
        self.to_k2 = torch.nn.Linear(self.attention_size * 2, self.attention_size)
        self.to_v2 = torch.nn.Linear(self.attention_size * 2, output_size)
        self.layernorm = torch.nn.LayerNorm(self.input_size)

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
        '''
        Arg(s):
            K : torch.Tensor[float32]
                N x H x W x C tensor
        Returns:
            torch.Tensor[float32] : N x C x H x W x k
        '''

        K = K.permute(0, 3, 1, 2)

        _, _, h, w = K.shape

        K_expanded = self.zero_padding(K)

        K = torch.unsqueeze(K, dim=-1).repeat(1, 1, 1, 1, self.kernel_size**2)

        pos = 0
        max_shift = int((self.kernel_size - 1) / 2)

        for i in range(-max_shift, max_shift + 1):
            for j in range(-max_shift, max_shift + 1):

                if i == 0 and j == 0:
                    pos += 1
                    continue

                height_start = i + max_shift
                height_end = height_start + h

                width_start = j + max_shift
                width_end = width_start + w

                K[:, :, :, :, pos] = K_expanded[:, :, height_start:height_end, width_start:width_end]
                pos += 1

        # 'K' has shape [batch, channel_size, h, w, neighbor_size]
        return K

    def compute_attention(self, K, Q):
        # 'Q' has shape [batch, h, w, attention_size]
        # 'K' has shape [batch, attention_size, h, w, neighbor_size]

        attention = torch.einsum('bhwa,bahwp->bphw', Q, K) * self.scale
#        attention = torch.einsum('bahwp,bahwp->bphw', Q, K) * self.scale
        attention = attention.softmax(dim=1) # + self.eps
        # 'attention' has shape [batch, neighbor_size, h, w]
        return attention

    def forward(self, x):
        # 'x' has shape [batch, input_size, h, w]
        pos_emb = self.create_positional_encoding()
        pos_emb = self.to_pos(pos_emb)
        pos_emb = self.ReLU(pos_emb)
        # 'pos_emb' has shape [neighbor_size, attention_size]

        x = x.permute(0,2,3,1)
        x = self.layernorm(x)

        K = self.to_k(x)
        K = self.ReLU(K)
        K = self.to_k2(K)

        K_no_pos = self.expand_shift(K)
        pos_k = self.to_posk(pos_emb).permute(1,0)
        # pos_k = self.to_k(pos_emb).permute(1,0)
        # bias_k = self.to_k(torch.zeros(self.input_size).to(device))
        K = K_no_pos + pos_k[None, :, None, None, :] #- bias_k[None, :,None, None, None]

        Q = self.to_q(x)
        Q = self.ReLU(Q)
        Q = self.to_q2(Q)

        V = self.to_v(x)
        V = self.ReLU(V)
        V = self.to_v2(V)

        V_no_pos = self.expand_shift(V)
        # pos_V = self.to_posv(pos_emb).permute(1,0)
        # pos_k = self.to_k(pos_emb).permute(1,0)
        # bias_k = self.to_k(torch.zeros(self.input_size).to(device))
        V = V_no_pos # + pos_V[None, :, None, None, :] #- bias_k[None, :,None, None, None]


        attention = self.compute_attention(K, Q)

        # 'attention' has shape [batch, neighbor_size, h, w]
        output = torch.einsum('bchwn,bnhw->bchw', V, attention)

        return output[:,:,::self.stride, ::self.stride]