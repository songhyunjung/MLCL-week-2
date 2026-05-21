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

def main():
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = False

    # 1. ClearML 태스크 초기화
    task = Task.init(project_name=Config.PROJECT_NAME, task_name=Config.TASK_NAME)
    logger = task.get_logger() # 실시간 그래프 출력을 위한 로거 객체 가져오기

    system = CaptioningSystem(Config)
    
    print("Loading datasets...")
    full_dataset = load_dataset(Config.DATASET_NAME, trust_remote_code=True)
    combined_data = full_dataset['train'] if 'train' in full_dataset else full_dataset['test']

    DEBUG_MODE = False
    if DEBUG_MODE:
        combined_data = combined_data.select(range(50))
        print(f"⚠️ DEBUG MODE ACTIVATED: Using {len(combined_data)} sample images.")

    dataset_split = combined_data.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = FlickrDataset(dataset_split["train"], system.tokenizer, system.preprocess, is_train=True)
    val_dataset = FlickrDataset(dataset_split["test"], system.tokenizer, system.preprocess, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=False)

    optimizer = optim.AdamW(system.model.clip_project.parameters(), lr=Config.LEARNING_RATE)

    print("Starting Fine-tuning (Mapping Network Only)...")
    global_step = 0 # 실시간 Step별 Loss 기록을 위한 인덱스
    
    for epoch in range(Config.EPOCHS):
        system.model.clip_project.train() 
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            loss = system.get_loss(batch["image"], batch["tokens"], batch["mask"])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # [추가] 매 배치의 실시간 훈련 Loss를 ClearML 대시보드 그래프에 플롯
            logger.report_scalar(
                title="Training Progress", 
                series="Batch Loss", 
                iteration=global_step, 
                value=loss.item()
            )
            global_step += 1
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
        
        # [추가] 에폭별 평균 Loss 추이 기록
        logger.report_scalar(
            title="Epoch Summary", 
            series="Avg Loss", 
            iteration=epoch + 1, 
            value=avg_epoch_loss
        )

    print("Training finished! Starting evaluation...")
    system.model.eval()
    
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(Config.DEVICE)
            
            prefix = system.clip_model.encode_image(images.type(system.clip_model.dtype)).to(torch.float32)
            prefix_embeds = system.model.clip_project(prefix) 
            
            if prefix_embeds.ndim == 2:
                batch_size = prefix_embeds.shape[0]
                prefix_embeds = prefix_embeds.view(batch_size, Config.PREFIX_LENGTH, -1)

            prefix_mask = torch.ones(prefix_embeds.shape[0], Config.PREFIX_LENGTH).to(Config.DEVICE)
            
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
    print(f"BLEU Score: {bleu:.4f}")
    print(f"CIDEr Score: {cider:.4f}")
    
    # [수정] 대시보드 상단의 '단일 요약 값(Summary)'과 '최종 그래프(Plots)' 둘 다 완벽하게 찍히도록 연동
    task.get_logger().report_single_value("Final BLEU", bleu)
    task.get_logger().report_single_value("Final CIDEr", cider)
    
    logger.report_scalar(title="Evaluation Metrics", series="BLEU", iteration=1, value=bleu)
    logger.report_scalar(title="Evaluation Metrics", series="CIDEr", iteration=1, value=cider)

if __name__ == "__main__":
    main()