import torch
from torch.utils.data import Dataset
import random

class FlickrDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, preprocess, is_train=True):
        """
        이미지 단위를 유지하여 Train/Val 분리 시 데이터 누수(Leakage)를 원천 차단합니다.
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.preprocess(item["image"].convert("RGB"))
        
        # 학습 시에는 5개의 정답 캡션 중 무작위로 1개를 선택하여 다양성 확보 (오버피팅 방지)
        # 평가 시에는 첫 번째 캡션을 기본 타깃으로 사용
        if self.is_train:
            caption = random.choice(item["caption"])
        else:
            caption = item["caption"][0]
        
        tokens = self.tokenizer(
            caption, 
            truncation=True, 
            max_length=40, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return {
            "image": image,
            "tokens": tokens["input_ids"].squeeze(0),
            "mask": tokens["attention_mask"].squeeze(0),
            "raw_caption": caption,
            "all_captions": item["caption"] # 평가 스코어 연산용 정답 리스트 5개 전체 유지
        }