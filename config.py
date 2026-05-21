import os
import torch

class Config:
    # 사용할 GPU 번호 세팅
    GPU_ID = 0
    DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    
    # ClearML 프로젝트 환경 설정
    PROJECT_NAME = "MLCL_Week2"
    TASK_NAME = "ClipCap_MAPPER"
    
    # Hugging Face 모델 및 데이터셋 고유 명칭
    MODEL_NAME = "michelecafagna26/clipcap-base-captioning-ft-hl-scenes"
    DATASET_NAME = "nlphuji/flickr30k"
    
    # 학습 하이퍼파라미터
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    EPOCHS = 15         # 성능 향상을 위해 10 Epoch 이상 권장
    MAX_LENGTH = 40
    PREFIX_LENGTH = 10  # CLIP에서 추출하여 GPT-2에 주입할 이미지 텍스트 프리픽스 길이
    
    # 가중치 저장 폴더 경로
    CHECKPOINT_DIR = "checkpoints"

# 저장용 디렉토리 자동 생성
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)