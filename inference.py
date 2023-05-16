
import os


from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch
device = 'cuda'
checkpoint = "THUDM/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained("/home/data/ChatGLM/model", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/data/ChatGLM/model", trust_remote_code=True).half().cuda()
model.eval()
LoRA=True
if LoRA:
    model_path='saved/finetune_test2'
    file=os.listdir(model_path)[2]
    print(file)
    # lora_config = { 
    #         'r': 32,
    #         'lora_alpha':32,
    #         'lora_dropout':0.1,
    #         'enable_lora':[True, True, True],
    #     }

    import loralib as lora
    from lora_utils.insert_lora import get_lora_model

    lora_config = {
        'r': 32,
        'lora_alpha':32,
        'lora_dropout':0.05,
        'enable_lora':[True, False, True],
    }

    model = get_lora_model(model, lora_config)


    model.load_state_dict(torch.load('saved/finetune_test/'+file), strict=False)
    model.half().cuda()
    model.eval()

device = 'cuda'

import dataset.GLM 
from torch.utils.data import DataLoader

dataset.GLM.device = device
#dataset.GLM.pad_to = 8

import dataset.Alpaca

import random
test_pairs= dataset.Alpaca.load('./data/LCCC-base_test.json')
random.shuffle(test_pairs)
completion=[i['completion'] for i in test_pairs]
pairs_encoded = dataset.GLM.encode_pairs(test_pairs, tokenizer, with_eos=False)
test_dataset = dataset.GLM.SimpleDataset(pairs_encoded)
BATCH_SIZE=10
test_dataloader = DataLoader(dataset=test_dataset, collate_fn = dataset.GLM.collate_fn, shuffle=False, batch_size=BATCH_SIZE)
len(test_dataloader)

import json
import tqdm
import re
result=[]
pbar = tqdm.tqdm(total=len(test_dataloader))
with open('result_'+file[-4]+'new.json', 'w', encoding='utf-8') as f:
# with open('result_new.json', 'w', encoding='utf-8') as f:
    for i,batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs=model.generate(
        **batch, 
        max_length=1024,
        eos_token_id=130005,
        do_sample=True,
        temperature=0.8,
        top_p = 0.75,
        top_k = 10000,
        repetition_penalty=1.5, 
        num_return_sequences=1,
        )
        templ=[{'completion':completion[i*BATCH_SIZE+j:i*BATCH_SIZE+j+1][0]} for j in range(BATCH_SIZE) if completion[i*BATCH_SIZE+j:i*BATCH_SIZE+j+1]]
        
        for j,output in enumerate(outputs):
            pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9。〜，！？“”‘’,."!?\[\]@#$%&*()（）【】<>《》{}~\-——+=|:;；：、|]'
            try:
                # text=tokenizer.decode(output).split('###Response:')[-1]
                text = re.sub(pattern, '', tokenizer.decode(output).split('Response:')[-1])
            except:
                text=''
            # text = re.sub(pattern, '', tokenizer.sp_tokenizer.decode(output).split('你的回答: ')[-1])
            templ[j]['response']=text
        result+=templ
        pbar.update(1)
    f.write(json.dumps(result, ensure_ascii=False, indent=4))
    pbar.close()

import requests
headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjExNTI0NiwidXVpZCI6ImQwNWViMzA0LWZkOGYtNGQ0ZC05MTJiLWI1OWZjZDE2ZTRhYyIsImlzX2FkbWluIjpmYWxzZSwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.e3nE8Wm3D2IqK4qKZwrV7B1Sxex0Jvp-hi9cBEC42tT1L5ZppkcXjb-u4nUTiwIOokDkv_W6IaxdshgM_wwo9w"}

test_accuracy=0
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                    json={
                        "title": "eg. 来自我的程序",
                        "name": "eg. 我的fine_tune实验",
                        "content": "实验结束",
                    }, headers = headers)
print(resp.content.decode())