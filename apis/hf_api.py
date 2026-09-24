import torch
import numpy as np
from tqdm import tqdm
import logging
from .api import API
import transformers
import random
from .utils import set_seed, get_subcategories, ALL_styles
import re
import collections


class HFAPI(API):
    def __init__(self,
                 model_name, variation_type, use_subcategory,
                 output_dir, seed, mlm_probability,
                 length, temperature, top_k, top_p, repetition_penalty, do_sample, fp16,
                 no_cuda, num_beams, dry_run,
                 variation_batch_size,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.variation_type = variation_type
        self.output_dir = output_dir
        self.length = length
        self.temperature = temperature
        self.k = top_k
        self.p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.fp16 = fp16
        self.no_cuda = no_cuda
        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        set_seed(seed=seed, n_gpu=self.n_gpu)
        self.dry_run = dry_run

        model_name_or_path = self.model_name

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if "gpt2" not in self.model_name:
            # use torch.float16 for large LLMs
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto", torch_dtype=torch.float16)
        else:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto", pad_token_id=pad_token_id)
            if self.fp16:
                self.model.half()

        self.variation_batch_size = variation_batch_size

    @staticmethod
    def command_line_parser():
        parser = super(HFAPI, HFAPI).command_line_parser()
        parser.add_argument(
            '--model_name',
            type=str,
            default='Qwen/Qwen2.5-1.5B-Instruct',
            help='Which image feature extractor to use')
        parser.add_argument("--use_subcategory",
                            action="store_true", help="use subcategory")
        parser.add_argument(
            '--variation_type',
            type=str,
            default='rephrase',
            choices=["imdb_rephrase", "openreview_rephrase_tone", "pubmed_rephrase_tone", "sst5_rephrase",
                     ],
            help='Which image feature extractor to use')
        parser.add_argument("--mlm_probability", type=float, default=0.5)

        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
        )
        parser.add_argument("--length", type=int, default=512)
        parser.add_argument("--temperature", type=float, default=1.0,)
        parser.add_argument("--repetition_penalty", type=float, default=1.0,
                            help="primarily useful for CTRL model; in that case, use 1.2")
        parser.add_argument("--top_k", type=int, default=50)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--num_beams", type=int, default=5)
        parser.add_argument("--do_sample", type=bool, default=True,
                            help="sampling when generation")
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument("--dry_run", action="store_true", help="dry run")
        parser.add_argument(
            '--variation_batch_size',
            type=int,
            default=256,
            help='The batch size for variation API')

        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument("--no_cuda", action="store_true",
                            help="Avoid using CUDA when available")

        return parser


    def text_variation(self, samples, labels,
                       num_variations_per_sample, variation_type, keep_original=True):
        if num_variations_per_sample == 0:
            raise ValueError(
                "num_variations_per_sample should be greater than 0")
        self.model.eval()
        self.model.to(self.device)
        variations = []
        lbs = []
        for idx in tqdm(range(num_variations_per_sample)):
            sub_variations, var_labels = self._text_variation(
                sequences=samples,
                labels=labels,
                variation_type=variation_type,
                batch_size=self.variation_batch_size)
            variations.append(sub_variations)
            lbs.append(labels)
        if keep_original:
            variations.append(samples)
            lbs.append(labels)
        torch.cuda.empty_cache()
        sample_stacked = np.stack(variations, axis=1)
        label_stacked = np.stack(lbs, axis=1)
        all_samples = []
        all_labels = []
        for i in range(len(sample_stacked)):
            all_samples.extend(sample_stacked[i])
            all_labels.extend(label_stacked[i])
        return all_samples, all_labels

    def _rephrase(self, label, sequence, variation_type):

        if variation_type == "imdb_rephrase":
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            prompt = "Based on {}, please rephrase the following sentences {} as a movie review, please make it fluent and natural:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "openreview_rephrase":
            selected_style = ALL_styles[random.randrange(
                len(ALL_styles))]
            prompt = "Based on {}, please rephrase the following sentences {} as a paper review, please make it fluent and natural:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "sst5_rephrase":
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            prompt = "Based on the sentiment {}, please rephrase the following sentences {}, please make it fluent and natural:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "yelp_rephrase":
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            prompt = "Based on {}, please rephrase the following sentences {}:\n{} \n".format(
                label, selected_style, sequence)
        return prompt

    def _text_variation(self, sequences, labels, variation_type, batch_size):
        num_seq = len(sequences)
        all_data = []
        all_labels = []
        self.model.eval()

        for i in tqdm(range(num_seq // batch_size + 1)):
            start_idx = i*batch_size
            if start_idx >= num_seq:
                break
            end_idx = num_seq if (
                i+1)*batch_size > num_seq else (i+1)*batch_size

            batch_prompt = []
            batch_labels = []
            for idx in range(start_idx, end_idx):
                prompt = self._rephrase(
                    labels[idx], sequences[idx], variation_type)
                batch_prompt.append(prompt)
                batch_labels.append(labels[idx])

            with torch.no_grad():
                input_ids = self.tokenizer(batch_prompt, padding=True, return_tensors='pt').to(self.device)  # has been padded into the same lens; cannot be used
                attention_mask = input_ids.attention_mask
                input_ids = input_ids.input_ids
                beam_output = self.model.generate(input_ids,
                                                  max_new_tokens=self.length,
                                                  temperature=self.temperature,
                                                  top_p=self.p,
                                                  pad_token_id=self.tokenizer.eos_token_id,
                                                  do_sample=self.do_sample,
                                                  attention_mask=attention_mask,
                                                  )
                generated_sequences = self.tokenizer.batch_decode(
                    beam_output[:, input_ids.shape[1]:], skip_special_tokens=True,  clean_up_tokenization_spaces=True)
            for idx in range(len(generated_sequences)):
                seq = generated_sequences[idx]
                seq = " ".join(seq.split())
                lab = batch_labels[idx].strip().split("\t")
                if seq:
                    all_data.append(seq)  # no lables!
                else:
                    all_data.append(batch_prompt[idx])
                all_labels.append(lab)

        logging.info(f" _text_variation output lens  {len(all_data)}")

        return all_data, all_labels
    def text_random_sampling(self, num_samples, prompt_counter=None, variation_type = None, lens_dict=None):
        ratio_generation_training = 1
        all_sequences = []
        ppls_cur = []
        additional_info = []
        sync_labels_counter = collections.Counter()

        self.model.eval()

        simulate_num = 0
        for prompt in tqdm(prompt_counter):
            # generation is proportional to the label distributions
            simulate_num_seq_to_generate = round(
                prompt_counter[prompt] * ratio_generation_training)
            simulate_num += simulate_num_seq_to_generate

        logging.info(
            f"should -- simulated generated sequences: %d", simulate_num)
        all_prefix_prompts = []
        for prompt in tqdm(prompt_counter):
            # generation is proportional to the label distributions
            num_seq_to_generate = round(
                prompt_counter[prompt] * ratio_generation_training)
            if "sst5" in variation_type:
                full_prompt_text = f"Suppose you are a sentiment expert, write a review based on sentiment {prompt}"
            elif "yelp" in variation_type:
                full_prompt_text = f"Suppose you are a review expert, write a review based on {prompt}"
            elif "openreview" in variation_type:
                full_prompt_text = f"Suppose you are a research paper expert, write a review based on {prompt}"
            elif "imdb" in variation_type:
                full_prompt_text = f"Suppose you are a movie review expert, write a review based on {prompt}"

            prompt_input_ids = self.tokenizer(full_prompt_text)['input_ids']
            before_gen_length = len(full_prompt_text)

            if num_seq_to_generate > 0:
                # condition on the prompt
                sequences = self._generate_text(prompt_input_ids, num_seq_to_generate,
                                                max_length=self.length, batch_size=self.variation_batch_size,
                                                before_gen_length=before_gen_length)
                all_sequences += sequences
            all_prefix_prompts += [full_prompt_text] * num_seq_to_generate
            additional_info += [prompt] * num_seq_to_generate
            sync_labels_counter[prompt] = num_seq_to_generate

        logging.info(f"Total generated sequences: %d", len(all_sequences))
        torch.cuda.empty_cache()
        return all_sequences,  additional_info, sync_labels_counter, all_prefix_prompts

    def _generate_text(self, prompt, seq_num, max_length, batch_size, before_gen_length):

        all_data = []

        if seq_num < batch_size:
            batch_size = seq_num + 1

        num_return_sequences = 2 if batch_size > 1 else 1
        for i in tqdm(range(seq_num // batch_size + 1)):
            if self.dry_run:
                generated_sequences = ["s" * max_length] * batch_size
            else:
                input_ids = torch.tensor(prompt).repeat(
                    batch_size, 1).to(self.device)
                with torch.no_grad():
                    output_sequences = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_length,
                        temperature=self.temperature,
                        top_p=self.p,
                        early_stopping=True,
                        do_sample=self.do_sample,
                        # overgenerate to ensure we have enough non-empty generated sequences
                    )
                    generated_sequences = self.tokenizer.batch_decode(output_sequences[:, input_ids.shape[1]:], skip_special_tokens=True,
                                                                      clean_up_tokenization_spaces=True)
            for g in generated_sequences:
                seq = g
                seq = " ".join(seq.split())
                if seq:
                    all_data.append(seq)

        if len(all_data) > seq_num:
            all_data = random.sample(all_data, seq_num)
        return all_data