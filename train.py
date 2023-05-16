
import os
import time
import tqdm
import json
import torch
import numpy as np
import loralib as lora
from lora_utils.insert_lora import get_lora_model


from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup



checkpoint = "THUDM/chatglm-6b"

model_id = "finetune_test"

mixed_precision = 'bf16'
lora_config = {
    'r': 32,
    'lora_alpha':32,
    'lora_dropout':0.1,
    'enable_lora':[True, False, True],
}

LR = 1e-4
BATCH = 1
MAX_LENGTH = 256
NUM_EPOCHS = 3
accumulate_step = 8
warm_up_ratio = 0.1



deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, deepspeed_plugin=deepspeed_plugin, log_with="tensorboard", project_dir='runs/')
device = accelerator.device


with accelerator.main_process_first():
    retry_cnt = 10
    cnt = 0
    while cnt < retry_cnt:
        try:
            import dataset.GLM
            tokenizer = AutoTokenizer.from_pretrained("/home/data/ChatGLM/model", trust_remote_code=True)
            model = AutoModel.from_pretrained("/home/data/ChatGLM/model", trust_remote_code=True)
            if mixed_precision == None:
                model = model.float()
            break
        except:
            cnt += 1 

    model = get_lora_model(model, lora_config)
    # model.load_state_dict(torch.load('saved/chatglm-6b_alpaca_5.pt'), strict=False)


accelerator.wait_for_everyone()

model.use_cache = False
model.gradient_checkpointing = False


import dataset.Alpaca as Alpaca_Data
dataset.GLM.device = device


accelerator.print('Start to process data')



with accelerator.main_process_first():
    pairs = Alpaca_Data.load('./data/train_alpaca.json')
    pairs_encoded = dataset.GLM.encode_pairs(pairs, tokenizer)
    pairs_encoded = list(filter(lambda pair: len(pair['prompt'])+len(pair['completion']) <= MAX_LENGTH, pairs_encoded))
train_dataset = dataset.GLM.SimpleDataset(pairs_encoded)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn = dataset.GLM.collate_fn, shuffle=True, batch_size=BATCH)



accelerator.wait_for_everyone()



optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(len(train_dataloader) // accumulate_step * NUM_EPOCHS),
)
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)




accelerator.init_trackers(model_id, {})

total_effective_step = 0

for epoch in range(NUM_EPOCHS):

    batch_loss = 0
    effective_step = 0
    
    for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):

        outputs = model(**batch)

        loss_d = outputs.loss.detach().cpu().float().item()
        batch_loss += loss_d

        loss = outputs.loss / accumulate_step
        accelerator.backward(loss)

        if (step+1) % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            effective_step += 1

            gathered_batch_loss = accelerator.gather((torch.tensor(batch_loss, device=device)))

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "train_loss": gathered_batch_loss.mean().item() / accumulate_step,
                        "epoch": epoch,
                    },
                    step = total_effective_step + effective_step,
                )

            t.set_description(f"loss: {gathered_batch_loss.mean().item() / accumulate_step}")
            batch_loss = 0   
        
    
    accelerator.wait_for_everyone()
    
    total_effective_step += effective_step
    
    if accelerator.is_main_process:
        os.makedirs(f'saved/{model_id}', exist_ok = True)
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), f'saved/{model_id}/{model_id}_epoch_{epoch}.pt')

    accelerator.wait_for_everyone()
