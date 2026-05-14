import torch
from casualalign import CasualAligner

class BatchedCasualAligner(CasualAligner):
    def __init__(self, model_size="base", device=None):
        super().__init__(model_size=model_size, device=device)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def align_batch(self, src_texts, tgt_texts):
        if len(src_texts) != len(tgt_texts):
            raise ValueError("Source and target text lists must have the same length")
        
        src_encodings = self.tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        tgt_encodings = self.tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            src_embeds = self.model.get_input_embeddings()(src_encodings.input_ids)
            tgt_embeds = self.model.get_input_embeddings()(tgt_encodings.input_ids)
            
            src_attn = src_encodings.attention_mask
            tgt_attn = tgt_encodings.attention_mask
            
            align_scores = self.compute_alignment_scores(src_embeds, tgt_embeds, src_attn, tgt_attn)
        
        return align_scores
    
    def compute_alignment_scores(self, src_embeds, tgt_embeds, src_attn, tgt_attn):
        src_embeds = src_embeds / src_embeds.norm(dim=-1, keepdim=True)
        tgt_embeds = tgt_embeds / tgt_embeds.norm(dim=-1, keepdim=True)
        
        batch_size = src_embeds.shape[0]
        src_len = src_embeds.shape[1]
        tgt_len = tgt_embeds.shape[1]
        
        src_embeds_flat = src_embeds.reshape(batch_size * src_len, -1)
        tgt_embeds_flat = tgt_embeds.reshape(batch_size * tgt_len, -1)
        
        scores = torch.matmul(src_embeds_flat, tgt_embeds_flat.T)
        scores = scores.reshape(batch_size, src_len, tgt_len)
        
        src_attn_exp = src_attn.unsqueeze(-1).expand(scores.shape)
        tgt_attn_exp = tgt_attn.unsqueeze(1).expand(scores.shape)
        scores = scores * src_attn_exp * tgt_attn_exp
        
        return scores