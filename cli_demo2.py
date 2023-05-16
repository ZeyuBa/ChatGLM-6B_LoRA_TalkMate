import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import torch
loRa=True
device = 'cuda'
checkpoint = "THUDM/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
model = AutoModel.from_pretrained("./model", trust_remote_code=True).half().cuda()
d={
"instruction":(
        "假设你是一名中国科学院大学（国科大）南京学院的2022级研究生，以下的聊天记录来自你与你的家人或朋友。"
        "请根据以下聊天记录的内容继续跟他聊天，双方的聊天用换行分开，你的回答要尽量简洁，口语化。\n\n"
        
)}
PROMPT_DICT = {
    "prompt_input": ( "Below is an instruction that describes a task, paired with an input that provides further context. "
                      "Write a response that appropriately completes the request.\n\n"
                      "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"              
    )
}
if loRa:
    import loralib as lora
    from lora_utils.insert_lora import get_lora_model

    lora_config = {
        'r': 32,
        'lora_alpha':32,
        'lora_dropout':0.05,
        'enable_lora':[True, False, True],
    }

    model = get_lora_model(model, lora_config)
    model.load_state_dict(torch.load('./finetune_test2/finetune_test_epoch_2.pt'), strict=False)
    model.half().cuda()
    model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(response,text):
    # prompt = "欢迎和来自UCASNJ的研究生xxx交流，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    prompt=''
    prompt += f"\n\n你：{text}"
    prompt += f"\n\nUCASNJ研究生xxx：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

def main():
    history = []
    global stop_stream
    print("欢迎和来自UCASNJ的研究生xxx交流，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    chat_history=''
    while True:
        # print('chat_history: \n',chat_history,end='')
        if len(chat_history.split('\n'))>6:
            chat_history=''
        text=input("\n你：")
        chat_history+=text+'\n'
        d['input'] = "### 聊天记录:\n"+chat_history
        query = PROMPT_DICT['prompt_input'].format_map(d)
        if "stop" in query:
            break
        if "clear" in query:
            history = []
            os.system(clear_command)
            print("欢迎和来自UCASNJ的研究生xxx交流，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            
            if stop_stream:
                stop_stream = False
                break
            # else:
            #     count += 1
            #     if count % 8 == 0:
            #         os.system(clear_command)
            #         print(build_prompt(history,text), flush=True)
            #         signal.signal(signal.SIGINT, signal_handler)
        chat_history+=response+'\n'
        os.system(clear_command)
        print('history: ',history)
        print(build_prompt(history[-1][-1],text), flush=True)


if __name__ == "__main__":
    main()
