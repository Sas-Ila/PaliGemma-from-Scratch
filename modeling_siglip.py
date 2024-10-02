from typing import Optional, Tuple
import torch
import torch.nn as nn



class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size= 768,
            intermediate_size= 3072,
            nb_hidden_layers= 12,
            nb_attention_heads= 12,
            nb_channels= 3,
            image_size= 224,
            patch_size= 16,
            layer_norm_eps= 1e-9,
            attention_dropout= 0,
            nb_image_tokens: int = None, 
            **kwargs 
                 ):
        
        super().__init__()
        self.hidden_size= hidden_size
        self.intermediate_size= intermediate_size
        self.nb_hidden_layers= nb_hidden_layers
        self.nb_attention_heads= nb_attention_heads
        self.nb_channels= nb_channels
        self.image_size= image_size
        self.patch_size= patch_size
        self.layer_norm_eps= layer_norm_eps
        self.attention_dropout= attention_dropout
        self.nb_image_tokens= nb_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
    
        self.patch_embedding = nn.Conv2d(
                            in_channels= config.nb_channels,
                            out_channels= self.embed_dim,
                            kernel_size= self.patch_size,
                            stride= self.patch_size,
                            padding='valid', # no padding added
        )
        self.nb_patches = (self.image_size // self.patch_size) ** 2
        self.nb_positions = self.nb_patches
        self.position_embedding = nn.Embedding(self.nb_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.nb_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width =  pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embds = self.patch_embedding(pixel_values)
        
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embds.flatten(2)

        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)

        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
    
class SiglipAttention(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.nb_heads  = config.nb_attention_heads

        assert self.embed_dim % self.nb_heads == 0, "The embedding dimension mus be a multiple of nb_heads."
        self.head_dim = self.embed_dim // self.nb_heads
        self.dropout = config.attention_dropout

        self.Wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wo = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.shape

        # [Batch_Size, Num_Patches, Embed_Dim]
        Q = self.Wq(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        K = self.Wk(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        V = self.Wv(hidden_states)

        # [Batch_Size, Num_Patches, Nb_Heads, Head_Dim]
        Q = Q.view(batch_size, seq_len, self.nb_heads, self.head_dim)
        Q = Q.transpose(1, 2) # [Batch_Size, Nb_Heads, Num_Patches, Head_Dim]

        K = K.view(batch_size, seq_len, self.nb_heads, self.head_dim)
        K = K.transpose(1, 2) # [Batch_Size, Nb_Heads, Num_Patches, Head_Dim]

        V = V.view(batch_size, seq_len, self.nb_heads, self.head_dim)
        V = V.transpose(1, 2) # [Batch_Size, Nb_Heads, Num_Patches, Head_Dim]

        # Calculate the attention weights matrix for each head; [Batch_Size, Nb_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(Q, K.transpose(2,3))/torch.sqrt(self.head_dim))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        attn_weights = nn.functional(attn_weights, p= self.dropout, training= self.training)

        # [Batch_Size, Nb_Heads, Num_Patches, Num_Patches] ->  [Batch_Size, Nb_Heads, Num_Patches, Head_Dim]
        attn_outputs = torch.matmul(attn_weights, V)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.reshape(batch_size, seq_len, self.embed_dim)
        return self.Wo(attn_outputs), attn_weights
    
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.FC1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.FC2 = nn.Linear(self.config.intermediate_size, self.config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_size]
        hidden_states = self.FC1(hidden_states)
        # [Batch_Size, Num_Patches, Intermediate_size] -> [Batch_Size, Num_Patches, Intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        # [Batch_Size, Num_Patches, Intermediate_size] -> [Batch_Size, Num_Patches, Embed_Dim] 
        hidden_states = self.FC2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return hidden_states
    
class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.LN1 = nn.LayerNorm(self.embed_dim, self.config.layer_norm_eps)
        self.MLP = SiglipMLP(config)
        self.LN2 = nn.LayerNorm(self.embed_dim, self.config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        # [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.LN1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        # [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.LN2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.MLP(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        # [Batch_Size, Num_Patches, Embed_Dim]
        return hidden_states
    


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoder(config) for _ in range(config.nb_hidden_layers)])
    
    def forward(self, input_embds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim]
        return hidden_states


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_LN = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]

        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embds= hidden_states)
        last_hidden_state = self.post_LN(last_hidden_state)

        return last_hidden_state



class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(self.config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values= pixel_values)



