import json
from nltk.translate.bleu_score import sentence_bleu
import jieba
def calculate_bleu_score(reference, candidate):
    reference_tokens = jieba.cut(reference)
    candidate_tokens = jieba.cut(candidate)
    
    # Convert the tokenized sentences into lists of tokens 
    reference_tokens = list(reference_tokens)
    candidate_tokens = list(candidate_tokens)
    
    # Calculate the BLEU score
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
    
    return bleu_score
b_l=[]
with open('result.json', 'r', encoding='utf-8') as f:
    result=json.load(f)
    for d in result:
        b_score=calculate_bleu_score(d['completion'],d['response'])
        b_l.append(b_score)
        # print(d['completion'],' '+d['response'],b_score) 
print('Mean BLEU score:', sum(b_l)/len(b_l))