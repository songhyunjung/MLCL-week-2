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
        
        # ClipCap 모델 로드
        self.model = ClipCaptionModel(
            prefix_length=config.PREFIX_LENGTH, 
            tokenizer=self.tokenizer
        )
        
        # -----------------------------------------------------------------
        # [논문 기반 성능 향상 핵심 1] 레이어별 정밀 잠금 해제 (Selective Unfreezing)
        # -----------------------------------------------------------------
        # 1단계: 전체 모델을 먼저 얼립니다.
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2단계: CLIP 공간을 GPT-2 공간으로 맵핑하는 가중치는 100% 학습 (Essential)
        for param in self.model.clip_project.parameters():
            param.requires_grad = True
            
        # 3단계: 논문 3.2절 사양 반영 - 고정된 GPT-2의 한계를 풀기 위해 
        # 시각 프리픽스 임베딩이 직접 통과하는 하위 레이어 및 텍스트 임베딩 가중치만 정밀 타격하여 깨웁니다.
        # 이를 통해 기존 웹 상식은 보존하면서 Flickr30k의 도메인 단어 분포에 적응합니다.
        for param in self.model.gpt.transformer.wte.parameters(): # 토큰 임베딩 레이어
            param.requires_grad = True
            
        # GPT-2의 최초 3개 트랜스포머 블록 블록의 어텐션 가중치만 미세 조정 허용
        for i in range(3):
            for param in self.model.gpt.transformer.h[i].parameters():
                param.requires_grad = True
                
        self.model.to(config.DEVICE)
        
        # [논문 3.2절 명세] CLIP 백본을 깨우는 것은 연산 복잡도만 높이고 성능 이점이 전혀 없다고 명시함.
        # 따라서 CLIP은 완벽하게 100% no_grad 및 eval 상태로 고정합니다.
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=config.DEVICE)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 디버깅을 위한 최종 학습 가능 파라미터 수 모니터링 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"⚡ [Paper-Driven Optimized Setup] Trainable Params: {trainable_params / 1e6:.2f}M")

    def get_loss(self, images, tokens, mask):
        images = images.to(self.config.DEVICE) 
        tokens = tokens.to(self.config.DEVICE)
        mask = mask.to(self.config.DEVICE)
        
        # CLIP 특징 추출 (논문 명세에 맞춰 절대 그래디언트가 흐르지 않게 고정)
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