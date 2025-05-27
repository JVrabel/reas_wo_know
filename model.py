import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast
from config import CONFIG

# Setup tokenizer
tok = GPT2TokenizerFast.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"

class TinyDec(nn.Module):
    def __init__(self, vocab=50257, d=CONFIG["MODEL_DIM"], n=CONFIG["MODEL_LAYERS"]):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(2048, d)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, 
            nhead=8, 
            dim_feedforward=4*d,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, n)
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        
        # Better initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, labels=None):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        x = self.emb(input_ids) + self.pos_emb(pos_ids)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(input_ids.device)
        
        # Transform
        x = self.transformer(x, x, tgt_mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1),
                ignore_index=-100
            )
            return loss, logits
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=10, temperature=1.0, do_sample=True, **kwargs):
        """Simple generation method."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we hit EOS
                if next_token.item() == tok.eos_token_id:
                    break
        
        return input_ids