import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
from tqdm import tqdm
import os
def intraSentenceDP(sentences, encoder_name='distilbert-base-uncased', decoder_name='bert-base-uncased', batch_size=16, eps=6, delta=1e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name).to(device)

    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    decoder = AutoModelForMaskedLM.from_pretrained(decoder_name).to(device)

    sigma = np.sqrt(2 * np.log(1.25 / delta)) / eps
    decoded_texts = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Processing"):
            batch = sentences[i:i + batch_size]
            encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
            embeddings = encoder(**encoded_input).last_hidden_state
            noisy_embeddings = embeddings + torch.normal(0, sigma, size=embeddings.shape).to(device)
            logits = decoder.cls(noisy_embeddings)
            predicted_ids = torch.argmax(logits, dim=-1)
            texts = decoder_tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            decoded_texts.extend(texts)

    return decoded_texts


if __name__ == '__main__':
    sentences = [
        "Hello, how are you doing?",
        "This is an example sentence.",
        "Differential privacy is useful in NLP."
    ]

    noisy_texts = intraSentenceDP(sentences,eps=12)

    for original, noisy in zip(sentences, noisy_texts):
        print(f"Original: {original}")
        print(f"Noisy: {noisy}")
        print('-' * 40)
