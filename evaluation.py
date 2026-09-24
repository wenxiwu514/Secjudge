from utils.tool import load_samples
import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import json
import transformers
import torch


def load_data(path='result/openreview/raw_data', num_samples=10):
    if 'imdb' in path:
        data_set = 'imdb'
    elif 'openreview' in path:
        data_set = 'openreview'
    elif 'sst5' in path:
        data_set = 'sst5'
    elif 'yelp' in path:
        data_set = 'yelp'
    samples, labels = load_samples(path, dataset=data_set)
    #if exist label set, load it
    with open('evaluation/dev_info/'+data_set+'.json', 'r') as f:
        dev_info = json.load(f)
    labelNum = len(labels[0].split('\t'))
    if num_samples < len(samples):
        #shuffle samples and labels in the same order
        sample_with_label = list(zip(samples, labels))
        np.random.seed(42)
        np.random.shuffle(sample_with_label)
        samples, labels = zip(*sample_with_label)
        samples = samples[:num_samples]
        labels = labels[:num_samples]
    # data = [dict(zip(['text'], [samples[i]])) for i in range(len(samples))]
    # data_0 = {'text': text, 'label':{}}
    data = [dict(zip(['text', 'label'], [samples[i], {}])) for i in range(len(samples))]
    for i in range(labelNum):
        for(idx, label) in enumerate(labels):
            lb = label.split('\t')[i]
            data[idx]['label']['original_'+str(i)] = lb
    return data, dev_info
def eval(client, data, dev_info, model='qwen-plus', path='', qpm=0):
    choice2labels = dev_info['labelSet']
    for i in range(len(choice2labels)):
        choice2label = choice2labels[str(i)]
        prompt_after = "The choices are: \n"
        for choice in choice2label.keys():
            prompt_after += choice + ": " + choice2label[choice] + "\n"
        prompt_before = dev_info['prompt_before'][str(i)]
        if qpm > 0:
            t1 = time.time()
        for item in tqdm(data, desc='eval:'+model+' '+path+' '+str(i)):
            prompt = prompt_before + item['text'] + "\n" + prompt_after
            completion = completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
            max_tokens=10,
            # stream_options={
            #     "include_usage": True
            # }
            )
            try:
                s = completion.choices[0].message.content
            except:
                s = "error"
            c = str.upper(s[0]) if s != "error" else "error"
            if c in choice2label.keys():
                item['label']['predict_'+str(i)] = choice2label[c]
            else:
                item['label']['predict_'+str(i)] = "error"
            item['response'] = s
            if qpm > 0:
                t2 = time.time()
                if t2 - t1 < 60/qpm:
                    time.sleep(60/qpm - (t2 - t1))
                t1 = t2
    return data

def eval_local(model, tokenizer, data, dev_info,model_name='', path='', batch_size = 16):
    choice2labels = dev_info['labelSet']
    cors = []
    all_probs = []
    for i in range(len(choice2labels)):
        choice2label = choice2labels[str(i)]
        prompt_after = "The choices are: \n"
        for choice in choice2label.keys():
            prompt_after += choice + ": " + choice2label[choice] + "\n"
        prompt_after += "The answer is: "
        prompt_before = dev_info['prompt_before'][str(i)]
        for batch_start in tqdm(range(0, len(data), batch_size), desc='eval:'+model_name+' '+path+' '+str(i)):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data[batch_start:batch_end]
            prompts = [prompt_before + item['text'] + "\n" + prompt_after for item in batch]


            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
            outputs = model.generate(**inputs, max_new_tokens=10)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_outputs = [s.strip() for s in decoded_outputs]
            for j, s in enumerate(decoded_outputs):
                c = str.upper(s[0]) if s != "error" else "error"
                if c in choice2label.keys():
                    batch[j]['label']['predict_'+str(i)] = choice2label[c]
                else:
                    batch[j]['label']['predict_'+str(i)] = "error"
                batch[j]['response'] = s
    return data
def evaluate_local(path, num_samples, model_name):

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if "Llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # if vicuna or 32B, use fp16
    if "vicuna" in model_name or "32B" in model_name:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.float16)
        batch_size = 8
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
        batch_size = 16
    data, dev_info = load_data(path, num_samples)
    data = eval_local(model, tokenizer, data, dev_info, model_name, path, batch_size)


    for i in range(len(dev_info['labelSet'])):
        acc = 0
        for item in data:
            if item['label']['original_'+str(i)] == item['label']['predict_'+str(i)]:
                acc += 1
        acc = acc / len(data)
        print('acc: ', acc)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    return data

def evaluate(api_key, base_url, path, num_samples, model, qpm=0):
    client = openai.OpenAI(
        api_key = api_key,
        base_url=base_url
    )
    data, dev_info = load_data(path, num_samples)
    data = eval(client, data, dev_info, model, path, qpm)
    for i in range(len(dev_info['labelSet'])):
        acc = 0
        for item in data:
            if item['label']['original_'+str(i)] == item['label']['predict_'+str(i)]:
                acc += 1
        acc = acc / len(data)
        print('acc: ', acc)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--base_url', type=str, default='')
    parser.add_argument('--path', type=str, default='', help='path to the data')
    parser.add_argument('--num_samples', type=int, default=1000, help='number of samples')
    parser.add_argument('--model', type=str, default='qwen-max', help='model name')
    parser.add_argument('--qpm', type=int, default=200, help='qpm')
    args = parser.parse_args()

    evaluate(args.api_key, args.base_url, args.path, args.num_samples, args.model, args.qpm)
