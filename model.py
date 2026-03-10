import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    embedding_dimension: int =512
    num_attention_heads: int=8
    attention_dropout_p: float=0.0
    hidden_dropout_p: float=0.0
    mlp_ratio: int=4
    encoder_depth : int=6
    decoder_depth : int=6

    src_vocab_size: int=30522 # bert tokenizer vocab size
    tgt_vocab_size: int =32000# vocab size of our italian tokenizer

    max_src_len: int=512 
    max_tgt_len: int= 512
    learn_pos_embed: bool=False

    

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim, requires_grad=False):
        super(PositionalEncoding,self).__init__()

        self.max_len= max_len
        self.embed_dim= embed_dim
        self.requires_grad= requires_grad

        self.encodings= self._build_positional_encodings()

    def _build_positional_encodings(self):
        encodings= torch.zeros(self.max_len, self.embed_dim, dtype=torch.float32)
        position_idx= torch.arange(0,self.max_len, dtype=torch.float32).reshape(-1,1)
        embed_skip_dim= torch.arange(0,self.embed_dim, step=2, dtype=torch.float32)

        encodings[:,0::2]= torch.sin(position_idx/ 10000**(embed_skip_dim/self.embed_dim)) # for even columns
        encodings[:,1::2]= torch.cos(position_idx/ 10000**(embed_skip_dim/self.embed_dim)) # for odd columns

        encodings=nn.Parameter(encodings, requires_grad=self.requires_grad)

        return encodings
    

    def forward(self,x):
        seq_len=x.shape[1]
        encodings= self.encodings[:seq_len]
        x=x+encodings
        return x    
    
class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings,self).__init__()

        self.src_embeddings= nn.Embedding(config.src_vocab_size, config.embedding_dimension)
        self.tgt_embeddings= nn.Embedding(config.tgt_vocab_size, config.embedding_dimension)
# We'll define two pos_encodings bcs it might be the case that u have different max lengths for two languages
        self.src_positional_encodings= PositionalEncoding(config.max_src_len,
                                                          config.embedding_dimension,
                                                          config.learn_pos_embed)


        self.tgt_positional_encodings= PositionalEncoding(config.max_tgt_len,
                                                          config.embedding_dimension,
                                                          config.learn_pos_embed)

    def forward_src(self,input_ids):
        embeddings= self.src_embeddings(input_ids)
        embeddings= self.src_positional_encodings(embeddings)# injecting positional encoding

        return embeddings

    def forward_tgt(self,input_ids):
        embeddings= self.tgt_embeddings(input_ids)
        embeddings= self.tgt_positional_encodings(embeddings)# injecting positional encoding

        return embeddings


       
class Attention(nn.Module):

    def __init__(self, config):
        super(Attention, self).__init__()
        
        self.config = config

        self.head_dim = config.embedding_dimension // config.num_attention_heads

        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        

    def forward(self, src, tgt=None, attention_mask=None, causal=False):
    
        batch, src_len, embed_dim = src.shape

        if tgt is None:
            q = self.q_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()

            if causal:
                # Build causal mask manually so we can combine it with padding mask
                causal_mask = torch.ones(src_len, src_len, dtype=torch.bool, device=src.device).tril()  # (src_len, src_len)
                if attention_mask is not None:
                    pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
                    combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & pad_mask  # (batch, 1, src_len, src_len)
                else:
                    combined_mask = causal_mask
                attention_out = F.scaled_dot_product_attention(q, k, v,
                                                            attn_mask=combined_mask,
                                                            dropout_p=self.config.attention_dropout_p if self.training else 0.0,
                                                            is_causal=False)  # False — causal handled via attn_mask
            else:
                if attention_mask is not None:
                    attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(1).repeat(1, 1, src_len, 1)
                attention_out = F.scaled_dot_product_attention(q, k, v,
                                                            attn_mask=attention_mask,
                                                            dropout_p=self.config.attention_dropout_p if self.training else 0.0,
                                                            is_causal=False)
        else:
            tgt_len = tgt.shape[1]
            q = self.q_proj(tgt).reshape(batch, tgt_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()

            if attention_mask is not None:
                attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(1).repeat(1, 1, tgt_len, 1)

            attention_out = F.scaled_dot_product_attention(q, k, v,
                                                        attn_mask=attention_mask,
                                                        dropout_p=self.config.attention_dropout_p if self.training else 0.0,
                                                        is_causal=False)

        attention_out = attention_out.transpose(1, 2).flatten(2)
        attention_out = self.out_proj(attention_out)
        return attention_out
        




class FeedForward(nn.Module):
    def __init__(self,config):
        super(FeedForward, self).__init__()

        hidden_size=config.embedding_dimension* config.mlp_ratio 
        self.intermediate_dense= nn.Linear(config.embedding_dimension, hidden_size)
        self.activation= nn.GELU()
        self.intermediate_drop= nn.Dropout(config.hidden_dropout_p)

        self.output_dense= nn.Linear(hidden_size, config.embedding_dimension)
        self.output_drop= nn.Dropout(config.hidden_dropout_p)

    def forward(self,x):
        x=self.intermediate_dense(x)
        x= self.activation(x)
        x=self.intermediate_drop(x)

        x= self.output_dense(x)
        x=self.output_drop(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder,self).__init__()

        self.encoder_attention=Attention(config)
        self.dropout= nn.Dropout(config.hidden_dropout_p)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)

        self.feed_forward= FeedForward(config)
        self.final_layer_norm= nn.LayerNorm(config.embedding_dimension)

    def forward(self,x,attention_mask=None):

        x= x+ self.dropout(self.encoder_attention(x,attention_mask= attention_mask,causal=False))
        x= self.layer_norm(x)

        x= x+ self.feed_forward(x)
        x= self.final_layer_norm(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()

        self.decoder_attention = Attention(config)
        self.decoder_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.decoder_attention_layer_norm = nn.LayerNorm(config.embedding_dimension)

        self.cross_attention = Attention(config)
        self.cross_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.cross_attention_layer_norm = nn.LayerNorm(config.embedding_dimension)

        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)

    def forward(self, encoder_out, tgt, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention on target sequence
        tgt = tgt + self.decoder_attention_dropout(
            self.decoder_attention(src=tgt, attention_mask=tgt_mask, causal=True)
        )
        tgt = self.decoder_attention_layer_norm(tgt)

        # 2. Cross-attention: queries from decoder, keys/values from encoder
        tgt = tgt + self.cross_attention_dropout(
            self.cross_attention(src=encoder_out, tgt=tgt, attention_mask=src_mask)
        )
        tgt = self.cross_attention_layer_norm(tgt)

        # 3. FFN
        tgt = tgt + self.feed_forward(tgt)
        tgt = self.final_layer_norm(tgt)

        return tgt

    

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer,self).__init__()

        self.config= config
        self.encodings= Embeddings(config)


        self.encoder=nn.ModuleList(
            [
                TransformerEncoder(config ) for i in range(config.encoder_depth  )
            ]
        )        

        self.decoder=nn.ModuleList(
            [
                TransformerDecoder(config ) for j in range(config.decoder_depth  )
            ]
        )      

        self.head=nn.Linear(config.embedding_dimension, config.tgt_vocab_size)
        self.apply(_init_weights_)

    def forward(self,src_ids, tgt_ids, src_attention_mask=None,tgt_attention_mask=None):

        src_embeddings= self.encodings.forward_src(src_ids)
        tgt_embeddings= self.encodings.forward_tgt(tgt_ids)

        for layer in self.encoder:
            src_embeddings= layer(src_embeddings,
                                src_attention_mask)
            

        for layer in self.decoder:
            tgt_embeddings=layer(src_embeddings, tgt_embeddings, src_attention_mask,tgt_attention_mask)


        pred= self.head(tgt_embeddings)
        return pred
    
    def inference(self, src_ids, tgt_start_id=2, tgt_end_id=3, max_len=8): # flag: will change max len
        with torch.no_grad():                 
            tgt_ids= torch.tensor([tgt_start_id], device=src_ids.device).reshape(1,1)
            src_embeddings= self.encodings.forward_src(src_ids)

            for layer in self.encoder:
                src_embeddings= layer(src_embeddings)

            for i in range(max_len-1): # we already started with [BOS] so we add -1

                tgt_embeddings= self.encodings.forward_tgt(tgt_ids)

                for layer in self.decoder:
                    tgt_embeddings= layer(src_embeddings,tgt_embeddings)

                tgt_embeddings= tgt_embeddings[:,-1] # we only need last timestamp, the last timestamp will have the context of previous timestamps

                pred=self.head(tgt_embeddings)
                pred= pred.argmax(axis=1).unsqueeze(0) # largest probability token

                tgt_ids= torch.cat([tgt_ids,pred], axis=-1)
                if torch.all(pred==tgt_end_id):
                    break
            
            return tgt_ids.squeeze().cpu().tolist()

@torch.no_grad()
def _init_weights_(module): # from hugginface implemetation_roberta.py, kind of initialization transformers like :)
    if isinstance(module, nn.Linear): 
        module.weight.data.normal_(0.0,0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(0.0,0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.fill_(1.0)
    


         
if __name__ == "__main__":  
    #pe=PositionalEncoding(512,512   )
    config= TransformerConfig()
    t=Transformer(config)

    english=torch.randint(low=0, high=1000,size=(1,32))
    italian=torch.randint(low=0, high=1000,size=(2,48))

                    
    t.inference(english)


