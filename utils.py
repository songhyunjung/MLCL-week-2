from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider

def calculate_metrics(all_preds, all_refs):
    """
    NLTK 및 COCO Evaluation Format 규칙에 맞게 차원을 전처리하여 스코어를 연산합니다.
    all_preds: [예측문장1, 예측문장2, ...]
    all_refs: [[정답1-1, 정답1-2, ..., 정답1-5], [정답2-1, ...], ...]
    """
    # BLEU 연산을 위해 단어 단위 토큰 스플릿 수행
    hypotheses = [p.split() for p in all_preds]
    references = [[r.split() for r in ref_list] for ref_list in all_refs]
    bleu = corpus_bleu(references, hypotheses)
    
    # CIDEr 스코어 연산 객체 빌드 및 실행
    cider_scorer = Cider()
    gts = {i: ref_list for i, ref_list in enumerate(all_refs)}
    res = {i: [p] for i, p in enumerate(all_preds)}
    cider, _ = cider_scorer.compute_score(gts, res)
    
    return bleu, cider