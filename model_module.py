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
        
        # [논문 수펙 반영 핵심 1] CLIP과 GPT-2는 완벽하게 얼립니다. (Frozen Encoders)
        for param in self.model.gpt.parameters():
            param.requires_grad = False
            
        # [논문 스펙 반영 핵심 2] 오직 가벼운 Mapping Network만 학습을 허용합니다.
        for param in self.model.clip_project.parameters():
            param.requires_grad = True
            
        self.model.to(config.DEVICE)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=config.DEVICE)

        # 변동 파라미터가 매핑 네트워크로 제한되므로 안전한 디버깅 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"⚡ [Paper Setup] Number of Trainable Parameters: {trainable_params / 1e6:.2f}M")

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