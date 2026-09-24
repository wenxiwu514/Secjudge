# SecJudge

> **An open-source framework for secure and verifiable LLM-as-a-judge services**

SecJudge is a pioneering platform designed for **secure and verifiable evaluation of large language models (LLMs)**, enabling researchers and developers to assess LLM outputs while protecting sensitive prompts and ensuring ownership verification. It leverages **differential privacy** and **clean-label watermarking** to balance privacy, utility, and detectability in LLM-as-a-judge scenarios.

## ✨ Key Features

- **Secure Prompt Obfuscation**: Implements the IS²-DP scheme for intra-sentence and inter-sentence shielding, protecting sensitive prompts using differential privacy.
- **Ownership Verification**: Employs clean-label watermarking with an optimized t-test detection protocol to identify unauthorized use of evaluation queries.
- **Unified Evaluation Pipeline**: Supports evaluation of LLMs across benchmark datasets (IMDb, SST-5, Yelp) with metrics like accuracy, Fréchet Inception Distance (FID), and F1 Score.

## 💾 Installation

```bash
git clone https://github.com/wenxiwu514/SecJudge.git
cd SecJudge

# (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Core dependencies: `torch`, `transformers`, `sentence_transformers`.

## 🚀 Quick Start

### 1. Generate Obfuscated Evaluation Queries

```bash
python main.py \
    --api hf \
    --dataset sst5 \
    --data_file data/sst5/sst5_test.csv \
    --word_embedding_model distilbert-base-uncased \
    --phrase_model Qwen/Qwen2.5-1.5B-Instruct \
    --combine_divide 4 \
    --epochs 1 \
    --num_private_samples 10 \ #test
    --result_folder result \
    --feature_extractor_batch_size 1024 \
    --feature_extractor all-mpnet-base-v2 \
    --noise_multiplier 0 \
    --nn_mode L2 \
    --count_threshold 0.0 \
    --select_syn_mode rank \
    --save_syn_mode selected \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --variation_batch_size 128 \
    --length 128
python watermarking.py \
    --sample_file path/to/sample_file.json \
    --validation_file watermark/validation.json \
    --query_file watermark/query.json \
    --trigger_word your_trigger_word \
```

This command applies the IS²-DP scheme to generate obfuscated queries with embedded watermarks.

### 2. Watermark Detection

```bash
python t-test.py \
```

### 3. Evaluate LLM Performance

```bash
python evaluation.py \
  --api_key "your_api_key" \
  --base_url "your_base_url" \
  --path "path/to/querysets" \
  --model "model_to_evaluate" \
```



## 🗂️ Project Layout

```
SecJudge/
├── scripts/              # Scripts for query generation, evaluation, and watermark detection
│   └── issdp.py
├── data/                 # Benchmark datasets (IMDb, SST-5, Yelp)
├── utils/                # Tools for differential privacy and watermarking
├── t-test.py             # Watermark detection script
├── evaluation.py         # Unified evaluation script
├── requirements.txt
└── README.md
```



## 📜 License

SecJudge is released under the MIT License. See the `LICENSE` file for details.

## 📞 Contact

For questions, please open an issue on [GitHub](https://github.com/wenxiwu514/Secjudge/issues).
