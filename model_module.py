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
        
        # [성능 향상 핵심] 라이브러리 내부에 걸려있는 GPT-2 가중치 동결을 강제로 전면 해제합니다.
        for param in self.model.gpt.parameters():
            param.requires_grad = True
            
        self.model.to(config.DEVICE)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=config.DEVICE)

        # 레이어별 차등 학습률(Layer-wise Learning Rate) 설정 인터페이스 구현
        # 새로 타깃 도메인을 배워야 하는 매핑 네트워크(clip_project)는 기존 학습률을 그대로 쓰고,
        # 이미 사전 학습이 잘 된 gpt 레이어는 상향된 LR의 50% 수준만 주어 텍스트 지식 붕괴를 예방합니다.
        self.optimizer_grouped_parameters = [
            {"params": self.model.clip_project.parameters(), "lr": config.LEARNING_RATE},
            {"params": self.model.gpt.parameters(), "lr": config.LEARNING_RATE * 0.5}
        ]

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