import torch
import clip
from clipcap import ClipCaptionModel
from transformers import GPT2Tokenizer
import torch.nn.functional as F

class CaptioningSystem:
    def __init__(self, config):
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = ClipCaptionModel(
            prefix_length=config.PREFIX_LENGTH, 
            tokenizer=self.tokenizer
        )
        
        # [수정 핵심 1] Mapping Network(통역사)와 GPT-2(언어 모델) 전체의 잠금을 완전히 해제합니다.
        for param in self.model.clip_project.parameters():
            param.requires_grad = True
            
        for param in self.model.gpt.parameters():
            param.requires_grad = True
            
        self.model.to(config.DEVICE)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=config.DEVICE)

    def get_loss(self, images, tokens, mask):
        images = images.to(self.config.DEVICE) 
        tokens = tokens.to(self.config.DEVICE)
        mask = mask.to(self.config.DEVICE)
        
        with torch.no_grad():
            prefix = self.clip_model.encode_image(images.type(self.clip_model.dtype)).to(torch.float32)
        
        prefix_mask = torch.ones(images.shape[0], self.config.PREFIX_LENGTH).to(self.config.DEVICE)
        combined_mask = torch.cat((prefix_mask, mask), dim=1)
        
        outputs = self.model(tokens, prefix, mask=combined_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        shift_logits = logits[:, self.config.PREFIX_LENGTH - 1 : -1, :].contiguous()
        shift_labels = tokens.contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id 
        )
        
        return loss