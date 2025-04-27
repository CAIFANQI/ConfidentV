import os
from openai import OpenAI
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from tqdm import tqdm
from typing import List, Dict
from dashscope import Generation


prompt_template = '''
Assume that signals are positive clock/clk edge triggered unless otherwise stated.
Implement the Verilog module based on the following description. 
Description:
{description}
{prompt}
'''


def get_model_output(question, m='qwen'):
    if m == 'qwen':
        c = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
               api_key="enter your api key")
        model = 'qwen2.5-7b-instruct'
    elif m == 'deepseek':
        c = OpenAI(api_key="enter your api key",
                           base_url="https://api.deepseek.com")
        model = 'deepseek-chat'
    else:
        c = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="enter your api key")
        model = 'doubao-1-5-pro-32k-250115'
    use_ft_model = True  # if don't use finetune model, set it False
    model = "qwen2.5-7b-instruct-ft-202504181242-10a8"

    max_retry = 3
    delay = 1
    retries = 0
    n_sample = 10

    cur_sample_num = 0
    samples = []
    while cur_sample_num < n_sample:
        while retries < max_retry:
            try:
                # fine-tuned model
                if use_ft_model:
                    response = Generation.call(
                        api_key="enter your api key",
                        model=model,
                        messages=[
                            {"role": "system", "content": "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with ’endmodule’. Do not include module, input and output definitions."},
                            {"role": "user", "content": question},
                        ],
                        result_format='message',
                        max_tokens=2048,
                        temperature=0.5,
                        stream=False,
                    )
                else:
                    response = c.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with ’endmodule’. Do not include module, input and output definitions."},
                            {"role": "user", "content": question},
                        ],
                        max_tokens=2048,
                        temperature=0.5,
                        stream=False,
                    )
            except Exception as e:
                retries += 1
                if retries == max_retry:
                    samples.append('')
                    retries = 0
                    cur_sample_num += 1
                    break
                time.sleep(delay)
                print(f"{model} get Error: {e} at {retries} tries. Retrying in {delay} seconds...")
            else:
                # different format
                if use_ft_model:
                    samples.append(response['output']['choices'][0]['message']['content'])
                else:
                    samples.append(response.choices[0].message.content)
                retries = 0
                cur_sample_num += 1
                break
    return samples


if __name__ == "__main__":
    prompt_type = ['Human', 'Machine']
    model_type = ['deepseek', 'doubao', 'qwen']
    for ty in prompt_type:
        description_file = f"VerilogDescription_{ty}.jsonl"
        prompt_file = f"VerilogEval_{ty}.jsonl"
        with open(description_file) as f:
            description_ls: List[Dict] = [json.loads(i) for i in f.readlines()]
        with open(prompt_file) as f:
            prompt_ls: List[Dict] = [json.loads(i) for i in f.readlines()]

        # format: {task_id: [description_idx, prompt_idx]}
        idx_dict = {i['task_id']: [idx] for idx, i in enumerate(description_ls)}
        for idx, i in enumerate(prompt_ls):
            idx_dict[i['task_id']].append(idx)

        input_ls = [(key, prompt_template.format(description=description_ls[value[0]]['detail_description'],
                                                 prompt=prompt_ls[value[1]]['prompt']))
                    for key, value in idx_dict.items()]

        n = 1  # min(os.cpu_count(), len(input_ls)) // 4
        for mo in model_type:
            if mo == 'qwen' and ty == 'Machine':
                continue
            fout = open(f"Verilog_{ty}_{mo}_ft_output.jsonl", 'w')
            with ThreadPoolExecutor(max_workers=n) as executor:
                for idx, future in tqdm(enumerate(executor.map(get_model_output, [o[1] for o in input_ls],
                                                               [mo]*len(input_ls)))):
                    tmp_dict = {'task_id': input_ls[idx][0], f'{mo}_output': future}
                    fout.write(json.dumps(tmp_dict) + "\n")
                    fout.flush()
            fout.close()


