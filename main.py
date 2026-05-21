import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from clearml import Task
from datasets import load_dataset

from config import Config
from dataset import FlickrDataset
from model_module import CaptioningSystem
from utils import calculate_metrics

def main():
    # GPU 메모리 및 CuDNN 자원 최적화
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = False

    # 1. ClearML 태스크 및 로거 초기화
    task = Task.init(project_name=Config.PROJECT_NAME, task_name=Config.TASK_NAME)
    logger = task.get_logger()

    # 2. 멀티모달 시스템 로드 (논문 기반 레이어 잠금 제어가 반영된 상태)
    system = CaptioningSystem(Config)
    
    print("Loading datasets...")
    full_dataset = load_dataset(Config.DATASET_NAME, trust_remote_code=True)
    combined_data = full_dataset['train'] if 'train' in full_dataset else full_dataset['test']

    # 디버그 모드 설정 활성화 여부
    DEBUG_MODE = False
    if DEBUG_MODE:
        combined_data = combined_data.select(range(50))
        print(f"⚠️ DEBUG MODE ACTIVATED: Using {len(combined_data)} sample images.")

    # Train / Validation 데이터셋 엄격 분리 (9:1)
    dataset_split = combined_data.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = FlickrDataset(dataset_split["train"], system.tokenizer, system.preprocess, is_train=True)
    val_dataset = FlickrDataset(dataset_split["test"], system.tokenizer, system.preprocess, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=False)

    # -----------------------------------------------------------------
    # [성능 향상 핵심 1] 레이어별 차등 학습률 기법 정의
    # -----------------------------------------------------------------
    # 통역 레이어(clip_project)는 적극적인 도메인 매핑을 위해 기본 LR(5e-5) 적용
    # 민감한 언어 모델 백본(wte 레이어 및 하위 3개 블록)은 상식 오염을 막기 위해 10분의 1 속도(5e-6) 적용
    optimizer = optim.AdamW([
        {'params': system.model.clip_project.parameters(), 'lr': Config.LEARNING_RATE},
        {'params': system.model.gpt.transformer.wte.parameters(), 'lr': Config.LEARNING_RATE * 0.1},
        {'params': [p for i in range(3) for p in system.model.gpt.transformer.h[i].parameters()], 'lr': Config.LEARNING_RATE * 0.1}
    ])
    
    # -----------------------------------------------------------------
    # [성능 향상 핵심 2] 코사인 어닐링 스케줄러 결합
    # -----------------------------------------------------------------
    # 에폭이 진행됨에 따라 학습률을 코사인 파동 형태로 감소시켜 정밀 수렴을 유도합니다.
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-7)

    print("Starting Fine-tuning with Paper-Driven Layerwise Optimization...")
    global_step = 0
    
    for epoch in range(Config.EPOCHS):
        system.model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            loss = system.get_loss(batch["image"], batch["tokens"], batch["mask"])
            loss.backward()
            
            # [성능 향상 핵심 3] 급격한 그래디언트 폭발을 방지하는 클리핑 기법 필수 연결
            torch.nn.utils.clip_grad_norm_(system.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # ClearML 배치별 실시간 Loss 리포트
            logger.report_scalar(
                title="Training Progress", 
                series="Batch Loss", 
                iteration=global_step, 
                value=loss.item()
            )
            global_step += 1
            
        # 에폭 종료 후 코사인 스케줄러 한 스텝 전진 (학습률 감속 반영)
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
        
        # ClearML 에폭 요약 차트 리포트
        logger.report_scalar(title="Epoch Summary", series="Avg Loss", iteration=epoch + 1, value=avg_epoch_loss)
        
        # 스케줄러에 의해 동적으로 감소하는 현재 매핑 레이어의 LR 추이 추적
        current_mapper_lr = optimizer.param_groups[0]['lr']
        logger.report_scalar(title="Optimization", series="Mapper Learning Rate", iteration=epoch + 1, value=current_mapper_lr)

    print("Training finished! Starting evaluation...")
    system.model.eval()
    
    all_preds = []
    all_refs = []

    # -----------------------------------------------------------------
    # [평가 단계] 안정적인 문장 완성을 위한 글로벌 빔 서치(Beam Size=5)
    # -----------------------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(Config.DEVICE)
            
            prefix = system.clip_model.encode_image(images.type(system.clip_model.dtype)).to(torch.float32)
            prefix_embeds = system.model.clip_project(prefix) 
            
            if prefix_embeds.ndim == 2:
                batch_size = prefix_embeds.shape[0]
                prefix_embeds = prefix_embeds.view(batch_size, Config.PREFIX_LENGTH, -1)

            prefix_mask = torch.ones(prefix_embeds.shape[0], Config.PREFIX_LENGTH).to(Config.DEVICE)
            
            # 논문 추천 사양인 빔서치 5 결합 탐색
            generated_ids = system.model.gpt.generate(
                inputs_embeds=prefix_embeds, 
                attention_mask=prefix_mask, 
                max_length=Config.MAX_LENGTH,
                do_sample=False,              
                num_beams=5,                  
                eos_token_id=system.tokenizer.eos_token_id,
                pad_token_id=system.tokenizer.pad_token_id 
            )
            
            preds = system.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_preds.extend([p.strip() for p in preds])
            
            formatted_refs = list(zip(*batch["all_captions"]))
            for ref_tuple in formatted_refs:
                all_refs.append(list(ref_tuple))

    print(f"Validation Sync Check - Preds: {len(all_preds)}, Refs: {len(all_refs)}")
    bleu, cider = calculate_metrics(all_preds, all_refs)
    
    print(f"\n[Evaluation Results]")
    print(f"Final BLEU Score: {bleu:.4f}")
    print(f"Final CIDEr Score: {cider:.4f}")
    
    # 최종 지표 스칼라 및 단일 요약 대시보드 리포트 통합 등록
    logger.report_single_value("Final BLEU Summary", bleu)
    logger.report_single_value("Final CIDEr Summary", cider)
    
    logger.report_scalar(title="Evaluation Metrics", series="BLEU", iteration=1, value=bleu)
    logger.report_scalar(title="Evaluation Metrics", series="CIDEr", iteration=1, value=cider)

if __name__ == "__main__":
    main()