import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt 

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, pos_enc_type, max_len=1024):
        super(CustomMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.pos_enc_type = pos_enc_type

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        if pos_enc_type == "relative":
            self.max_len = max_len
            self.Er = nn.Parameter(torch.randn(max_len, self.depth))
            # Adjust skew operation
            self.register_buffer("zero_pad", torch.zeros((1, 1, 1, max_len)))

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)

        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)

        Q_split = self.split_heads(Q, batch_size)
        K_split = self.split_heads(K, batch_size)
        V_split = self.split_heads(V, batch_size)

        # code from  https://jaketae.github.io/study/relative-positional-encoding/
        if self.pos_enc_type == "relative":
            seq_len = Q.size(1)
            if seq_len > self.max_len:
                raise ValueError("Sequence length exceeds model capacity")

            # Compute relative positional encodings with skew operation
            Er = self.Er[:seq_len, :]
            QEr = torch.matmul(Q_split, Er.transpose(-2, -1))
            Srel = self.skew(QEr)
            output, attention_weights = scaled_dot_product_attention(Q_split, K_split, V_split, Srel, mask)
        else:
            output, attention_weights = scaled_dot_product_attention(Q_split, K_split, V_split, mask)

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output)

        return {'output': output, 'attention_weights': attention_weights, 'Q': Q, 'K': K, 'V': V}

    def skew(self, QEr):
        # Dynamically create zero padding based on the shape of QEr
        batch_size, num_heads, seq_len, _ = QEr.shape
        zero_pad = torch.zeros((batch_size, num_heads, seq_len, 1), device=QEr.device, dtype=QEr.dtype)
        
        padded_QEr = torch.cat([zero_pad, QEr], dim=-1)
        reshaped = padded_QEr.reshape(batch_size, num_heads, seq_len + 1, seq_len)
        Srel = reshaped[:, :, 1:].contiguous()
        return Srel

class CustomEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout, pos_enc_type, length):
        super(CustomEncoderBlock, self).__init__()

        self.self_attn = CustomMultiHeadAttention(d_model, num_heads, pos_enc_type, length)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward1 = nn.Linear(d_model, ffn_dim)
        self.feed_forward2 = nn.Linear(ffn_dim, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN: Layer norm applied first
        x_norm = self.layer_norm1(x)
        
        # Attention mechanism
        attn_result = self.self_attn(Q=x_norm, K=x_norm, V=x_norm, mask=mask)

        # Attention weights context length * context length dim * 1 (softmax value)
        attention_graph = (attn_result['attention_weights'])

        # Apply dropout to the attention output, then add the residual (input x)
        attn_output = self.dropout(attn_result['output'])
        attn_output += x  # Residual connection with original input

        # Pre-LN for MLP
        mlp_input_norm = self.layer_norm2(attn_output)

        # MLP / Feed-forward network
        ff_output = self.feed_forward1(mlp_input_norm)
        ff_output_gelu = F.gelu(ff_output)  # Output after ReLU
        ff_output = self.feed_forward2(ff_output_gelu)

        # Apply dropout to the feed-forward output, then add the residual
        ff_output = self.dropout(ff_output)
        ff_output += attn_output  # Residual connection with the output of attention

        # Output dictionary
        output_dict = {
            'Q': attn_result['Q'],
            'K': attn_result['K'],
            'V': attn_result['V'],
            'attention_output': attn_output,
            'intermediate_residual_stream': x,
            'feed_forward_output_gelu': ff_output_gelu,
            'feed_forward_output': ff_output
            # 'attention_graph': attention_graph
        }

        return output_dict

def scaled_dot_product_attention(Q, K, V, pos_encodings, mask=None):
    matmul_qk = torch.matmul(Q, K.transpose(-2, -1))

    # only add if pos enc is relative 
    if isinstance(pos_encodings, torch.Tensor) and isinstance(matmul_qk, torch.Tensor):
        matmul_qk += pos_encodings

    d_k = Q.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)   

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        # pe shape from (1, max_len, d_model) sliced to (1, sequence_length, d_model)
        # Output shape will be same as x: (batch_size, sequence_length, d_model)
        return x + self.pe[:, :x.size(1), :]

class TweetyBERT(nn.Module):
    def __init__(self, d_transformer, nhead_transformer, num_freq_bins, num_labels, dropout=0.1, transformer_layers=3, dim_feedforward=128, m = 33, p = 0.01, alpha = 1, length = 500, pos_enc_type="relative"):
        super(TweetyBERT, self).__init__()
        self.num_labels = num_labels
        self.dropout = dropout
        self.m = m
        self.p = p 
        self.alpha = alpha 
        self.d_transformer = d_transformer
        self.pos_enc_type = pos_enc_type
        self.context = length 
        self.num_freq_bins = num_freq_bins

        # TweetyNet Front End
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))


        self.pos_enc = PositionalEncoding(d_transformer)
        # transformer
        self.transformerProjection = nn.Linear(768, d_transformer)
        self.transformer_encoder = nn.ModuleList([CustomEncoderBlock(d_model=d_transformer, num_heads=nhead_transformer, ffn_dim=dim_feedforward, dropout=dropout, pos_enc_type=pos_enc_type, length=length) for _ in range(transformer_layers)])        
        self.transformerDeProjection = nn.Linear(d_transformer, num_freq_bins)

        # Replace the hardcoded CUDA device with automatic detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def get_layer_output_pairs(self):
        layer_output_pairs = []

        # Create a dummy input for a complete forward pass
        ## temp fix 
        dummy_x = torch.randn(1, 1, self.d_transformer, device=self.device)

        # Extract the transformer forward outputs
        _, all_outputs = self.transformer_forward(dummy_x)

        # Iterate over each layer and its outputs
        for layer_index, output_dict in enumerate(all_outputs):
            for key in output_dict.keys():
                # Get the dimensionality of the output tensor
                dim = output_dict[key].size(-1)
                layer_output_pairs.append((key, layer_index, dim))

        return layer_output_pairs

    def feature_extractor_forward(self, x):
        x = F.gelu(self.conv1(x))
        x = self.pool1(x)
        x = F.gelu(self.conv2(x))
        x = self.pool2(x)
        x = F.gelu(self.conv3(x))
        x = self.pool3(x)
        x = F.gelu(self.conv4(x))
        x = self.pool4(x)

        x = x.flatten(1,2)

        return x

    def transformer_forward(self, x):
        all_outputs = []
        for layer in self.transformer_encoder:
            output_dict = layer(x) 
            all_outputs.append(output_dict)
            x = output_dict['feed_forward_output']  

        return x, all_outputs

    def masking_operation(self, x, p=0.01, m=10, noise_std=0.0):
        """
        First, apply noise augmentation to the input tensor `x` by adding normally distributed noise.
        Then, apply a mask to the augmented tensor with a total masked size of `m`, distributed randomly across the tensor.
        Each mask size from 1 to `m` has an equal probability of being chosen.
        
        Parameters:
            x (torch.Tensor): Input tensor with shape [batch, dim, length]
            p (float): (Unused) Probability of masking a particular element
            m (int): Total number of elements to mask
            noise_std (float): Standard deviation of the noise to be added
            
        Returns:
            torch.Tensor: Tensor with noise augmentation and masked areas, same shape as `x`
            torch.Tensor: Mask tensor with shape [batch, dim, length]
        """
        batch, dim, length = x.size()

        # # Apply noise augmentation
        # noise = torch.randn_like(x) * noise_std
        # x_augmented = x + noise

        # Initialize mask with zeros
        mask = torch.zeros(batch, dim, length, device=x.device)

        total_masked = 0
        while total_masked < m:
            remaining = m - total_masked
            # Ensure the mask size does not exceed the remaining number of elements to mask
            mask_size = torch.randint(1, remaining + 1, (1,)).item()
            start_index = torch.randint(0, length - mask_size + 1, (1,)).item()
            mask[:, :, start_index:start_index + mask_size] = 1
            total_masked += mask_size

        mask_float = mask.float()

        # Apply mask to the augmented tensor
        x = x * (1 - mask_float)

        return x, mask_float
    
    def learned_pos_embedding(self, x):
        # learned pos encoding
        # First, generate a sequence of positional indices
        position_ids = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        # Then, use the embedding layer to get the positional encodings
        pos_enc = self.learned_pos_enc(position_ids)
        # Add the positional encodings to the input
        x = x + pos_enc
        return x 
    
    def train_forward(self, spec):
        intermediate_outputs = {}

        x = spec.clone()
        intermediate_outputs["input_spec"] = x.clone()


        x, mask = self.masking_operation(x.squeeze(1), p=self.p, m=self.m)
        intermediate_outputs["masking_operation"] = x.clone()

        x = self.feature_extractor_forward(x.unsqueeze(1))
        intermediate_outputs["feature_extractor_forward"] = x.clone()

        x = x.permute(0,2,1)
        x = self.transformerProjection(x)

        intermediate_outputs["transformerProjection"] = x.clone().permute(0,2,1)

        # double check this 
        if self.pos_enc_type == "sinusodal":
            #sin pos enc
            x = self.pos_enc(x)

        elif self.pos_enc_type == "learned_embedding":
            # learned pos encoding 
            x = self.learned_pos_embedding(x)

        # if relative or none do nothing because its handled elsewhere 
        elif self.pos_enc_type == "relative" or self.pos_enc_type == None:
            pass 

        intermediate_outputs["pos_enc"] = x.clone().permute(0,2,1)

        x, all_outputs = self.transformer_forward(x)

        intermediate_outputs1 = intermediate_outputs.copy()
        for i, layer in enumerate(all_outputs):
            # Add key-value pairs from dict2, modifying keys that overlap
            for key, value in layer.items():
                new_key = key
                # Modify the key if it already exists in dict1
                new_key = f"{key}_{i}"

                if value.dim() <= 3:
                    intermediate_outputs1[new_key] = value.permute(0,2,1)
        
        intermediate_outputs = intermediate_outputs1
        x = self.transformerDeProjection(x)
        intermediate_outputs["transformerDeProjection"] = x.clone().permute(0,2,1)

        return x, mask, spec, intermediate_outputs
    
    def inference_forward(self, spec):
        x = spec.clone()

        x= self.feature_extractor_forward(x)
        
        x = x.permute(0,2,1)
        x = self.transformerProjection(x)
        
        # double check this 
        if self.pos_enc_type == "sinusodal":
            #sin pos enc
            x = self.pos_enc(x)

        elif self.pos_enc_type == "learned_embedding":
            # learned pos encoding 
            x = self.learned_pos_embedding(x)

        # if relative or none do nothing because its handled elsewhere 
        elif self.pos_enc_type == "relative" or self.pos_enc_type == None:
            pass 

        x, all_outputs = self.transformer_forward(x)
        x = self.transformerDeProjection(x)
        return x, all_outputs



    def mse_loss(self, predictions, mask, spec, intermediate_layers, vocalization):
        epsilon = 1e-6
        alpha = self.alpha
        
        # Ensure all tensors are on the same device
        device = predictions.device
        mask = mask.to(device)
        vocalization = vocalization.to(device)
        
        spec = spec.squeeze(1).permute(0, 2, 1)
        
        # MSE Loss
        mse_loss = F.mse_loss(predictions, spec, reduction='none')
        
        # Create masks
        masked_indices = mask[:, 0, :] == 1.0
        vocalized_indices = vocalization.bool()
        
        vocalized_and_masked_indices = masked_indices & vocalized_indices
        vocalized_and_unmasked_indices = vocalized_indices & ~masked_indices
        
        # Calculate losses
        masked_loss = mse_loss[vocalized_and_masked_indices].mean() if vocalized_and_masked_indices.any() else torch.tensor(epsilon, device=device)
        unmasked_loss = mse_loss[vocalized_and_unmasked_indices].mean() if vocalized_and_unmasked_indices.any() else torch.tensor(epsilon, device=device)
        
        # Combine losses
        combined_loss = alpha * masked_loss + (1 - alpha) * unmasked_loss
        
        return combined_loss, masked_loss, unmasked_loss, mse_loss