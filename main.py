from config import Config
from dataset import FlickrDataset
from model_module import CaptioningSystem
from utils import calculate_metrics
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from clearml import Task
import torch
from itertools import islice
from transformers import get_linear_schedule_with_warmup

def main():
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = False

    task = Task.init(project_name=Config.PROJECT_NAME, task_name=Config.TASK_NAME)
    system = CaptioningSystem(Config)
    
    print("Loading datasets...")
    # split 인자를 명시하지 않고 전체 데이터셋을 완전하게 다운로드합니다.
    full_dataset = load_dataset(Config.DATASET_NAME, trust_remote_code=True)
    
    # 전처리 과정에서 키 명이 유실되는 버그를 막기 위해 하나의 단일 데이터 집합으로 결합 후 재할당
    combined_data = full_dataset['train'] if 'train' in full_dataset else full_dataset['test']

    # [질문 반영] 샘플만 돌려보고 구조를 고치기 위한 하이브리드 디버그 모드 설계
    DEBUG_MODE = False  # 검증 완료 후 본 실험 시 False로 수정하세요.
    if DEBUG_MODE:
        combined_data = combined_data.select(range(50)) # 50개 이미지 샘플만 고속 테스트
        print(f"⚠️ DEBUG MODE ACTIVATED: Using {len(combined_data)} sample images.")

    # 1. 90% 학습용, 10% 검증용 스플릿 수행 (순수 이미지 단위 분할로 Leakage 차단)
    dataset_split = combined_data.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    # 데이터셋 객체 생성 시 학습 모드(is_train) 명시 지정
    train_loader = DataLoader(
        FlickrDataset(train_dataset, system.tokenizer, system.preprocess, is_train=True), 
        batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        FlickrDataset(val_dataset, system.tokenizer, system.preprocess, is_train=False), 
        batch_size=Config.BATCH_SIZE, shuffle=False
    )

    total_steps = len(train_loader) * Config.EPOCHS
    warmup_steps = int(total_steps * 0.1) # 하위 10% 스텝 구간은 웜업 처리로 수렴 안정화 확보
    
    # 변경: 시스템 클래스 내부에서 정의한 차등 학습률 그룹 리스트를 바인딩합니다.
    optimizer = optim.AdamW(system.optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    print("Starting Fine-tuning...")
    for epoch in range(Config.EPOCHS):
        system.model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            loss = system.get_loss(
                batch["image"], 
                batch["tokens"], 
                batch["mask"]
            )
            loss.backward()
            optimizer.step()
            scheduler.step() # 스케줄러를 배치 단위로 업데이트하여 정교한 수렴 실현
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            current_step = (epoch * len(train_loader)) + pbar.n 
            task.get_logger().report_scalar(
                title="Loss", series="train", iteration=current_step, value=loss.item()
            )

        checkpoint_path = f"{Config.CHECKPOINT_DIR}/model_epoch_{epoch+1}.pt"
        torch.save(system.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    print("Training finished! Starting evaluation...")
    system.model.eval()
    
    all_preds = []
    all_refs = []

    
    EVAL_SAMPLES = 50 if DEBUG_MODE else len(val_dataset)
    eval_steps = max(1, EVAL_SAMPLES // Config.BATCH_SIZE)

    with torch.no_grad():
        for batch in tqdm(islice(val_loader, eval_steps), total=eval_steps, desc="Evaluating"):
            images = batch["image"].to(Config.DEVICE)
            prefix = system.clip_model.encode_image(images.type(system.clip_model.dtype)).to(torch.float32)
            prefix_embeds = system.model.clip_project(prefix) 
            
            if prefix_embeds.ndim == 2:
                batch_size = prefix_embeds.shape[0]
                prefix_embeds = prefix_embeds.view(batch_size, Config.PREFIX_LENGTH, -1)

            prefix_mask = torch.ones(prefix_embeds.shape[0], Config.PREFIX_LENGTH).to(Config.DEVICE)
            
           # [수정] 가장 순수하고 왜곡 없는 정석 Greedy Search 알고리즘 명시
            generated_ids = system.model.gpt.generate(
                inputs_embeds=prefix_embeds, 
                attention_mask=prefix_mask, 
                max_length=Config.MAX_LENGTH,
                do_sample=False,                            # 확률 샘플링을 끄고 Greedy 모드 강제
                num_beams=1,                                # 빔 서치를 사용하지 않고 단일 최적 경로 탐색
                eos_token_id=system.tokenizer.eos_token_id,
                pad_token_id=system.tokenizer.pad_token_id 
            )
            
            preds = system.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # [보안] 예측 문장의 앞뒤 공백을 제거하여 텍스트 매칭 지표 연산 최적화
            preds = [p.strip() for p in preds]
            all_preds.extend(preds)
            
            # [오류 수정 핵심] 배치 크기(images.shape[0])에 맞춰 정답지 구조를 1:1로 정확하게 분해하여 적재
            # batch["all_captions"]의 형태는 [5(정답수), Batch_Size] 구조를 가지므로 차원을 뒤집어서 추출해야 합니다.
            for i in range(images.shape[0]):
                single_img_references = [caps[i] for caps in batch["all_captions"]]
                all_refs.append(single_img_references)

    # 최종 크기가 완벽하게 일치하는지 방어벽 로그 출력 (길이가 다르면 여기서 검증됨)
    print(f"Ref정렬 검증 - Predictions: {len(all_preds)}, References: {len(all_refs)}")

    bleu, cider = calculate_metrics(all_preds, all_refs)
    
    print(f"\n[Evaluation Results]")
    print(f"BLEU Score: {bleu:.4f}")
    print(f"CIDEr Score: {cider:.4f}")
    
    task.get_logger().report_single_value("Final BLEU", bleu)
    task.get_logger().report_single_value("Final CIDEr", cider)

if __name__ == "__main__":
    main()