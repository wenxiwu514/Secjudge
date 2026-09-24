import os
import numpy as np
from utils.feature_extractor import extract_features
from utils.intra_sentence_dp import intraSentenceDP
from utils.dp_nn_hist import dp_nn_histogram
from utils.metrics import compute_fid
from utils.tool import load_data, load_embeddings, save_samples, load_samples, parse_args
import logging
import collections




def main():
    args, api = parse_args()
    args.result_folder = os.path.join(args.result_folder, args.dataset)
    os.makedirs(args.result_folder, exist_ok=True)
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=os.path.join(args.result_folder, 'log.log'))
    logger = logging.getLogger(__name__)
    logger.info('load data')
    data_folder = os.path.join(args.result_folder, 'raw_data')
    all_private_samples, all_private_labels, private_labels_counter, private_labels_indexer = load_data(
        dataset=args.dataset,
        data_file=args.data_file,
        num_samples=args.num_private_samples)
    logger.info(f"Private_num_classes: {len(private_labels_counter)}, Private_num_samples: {len(all_private_samples)}")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        save_samples(all_private_samples, all_private_labels, data_folder, args.dataset)
    logging.info('load extract features')
    folder = os.path.join(args.result_folder, 'embeddings', args.feature_extractor)
    if not os.path.exists(folder):
        os.makedirs(folder)
    savefname = os.path.join(folder, f'{args.dataset}_{args.feature_extractor}_all.embeddings.npz')
    if os.path.exists(savefname):
        # if there is the embeddings file, load it
        logging.info(f'load features from {savefname}')
        all_private_features = load_embeddings(
            savefname)
    else:
        # extract the embeddings of the private data
        all_private_features = extract_features(
            data=all_private_samples,
            batch_size=args.feature_extractor_batch_size,
            model_name=args.feature_extractor,
        )
        np.savez(
            savefname,
            embeddings=all_private_features
        )
    print(f'features shape: {all_private_features.shape}, labels shape: {len(all_private_labels)}, num_data: {len(all_private_samples)}, num_classes: {len(private_labels_counter)}')
    private_classes = list(private_labels_counter.keys())

    # Generating intra-sentence DP data
    if args.method == 'issdp':
        logging.info('Generating intra-sentence DP data')
        savefolder = os.path.join(args.result_folder, 'intra_sentence_dp', args.word_embedding_model)
        if os.path.exists(savefolder):
            # if there is the intra-sentence DP data file, load it
            intraSdata, _ = load_samples(savefolder, args.dataset)
        else:
            intraSdata = intraSentenceDP(all_private_samples, args.word_embedding_model)
            save_samples(intraSdata, all_private_labels, savefolder, args.dataset)
    else:
        logging.info('Generating initial samples')
        savefolder = os.path.join(args.result_folder, 'initial_samples')
        if os.path.exists(savefolder):
            # if there is the initial samples file, load it
            seed_syn_samples, seed_additional_info = load_samples(savefolder, args.dataset)
        else:
            # generate the initial samples
            seed_syn_samples, seed_additional_info, sync_labels_counter, all_prefix_prompts = api.text_random_sampling(num_samples=args.num_private_samples, prompt_counter=private_labels_counter, variation_type = args.variation_type)
            save_samples(seed_syn_samples, all_private_labels, savefolder, args.dataset)
            save_samples(seed_syn_samples, seed_additional_info, savefolder, args.dataset, filename='seed_additional_info.json')
        intraSdata = seed_syn_samples

    # rephrase the intra-sentence DP data
    logging.info('Running text rephrase')
    print('Running text rephrase')
    args.result_folder = os.path.join(args.result_folder, args.phrase_model)
    savefolder = os.path.join(args.result_folder, 'rephrase_data')
    if os.path.exists(savefolder):
        # if there is the rephrase data file, load it
        rephrase_data, re_labels = load_samples(savefolder, args.dataset)
    else:
        # rephrase the intra-sentence DP data
        logging.info('Running text variation')
        rephrase_data, re_labels = api.text_variation(intraSdata, all_private_labels, num_variations_per_sample=1, variation_type=args.variation_type, keep_original=False)
        save_samples(rephrase_data, re_labels, f'{args.result_folder}/rephrase_data', args.dataset)
    print(len(rephrase_data), len(re_labels), len(all_private_labels))
    for i in range(len(re_labels)):
    # assert re_labels == all_private_labels
        # print(f're_labels: {re_labels[i]}, all_private_labels: {all_private_labels[i]}')
        assert re_labels[i] == all_private_labels[i], f're_labels: {re_labels[i]}, all_private_labels: {all_private_labels[i]}'

    # Generating init inter-sentence DP data
    logging.info('Generating init inter-sentence DP data')
    print('Generating init inter-sentence DP data')
    init_folder = os.path.join(args.result_folder, 'init_data')
    if not os.path.exists(init_folder):
        # syn_samples, syn_labels = [], []
        # sync_labels_counter = collections.Counter()
        # current_idx = 0
        # for class_i, class_ in enumerate(private_classes):
        #     print(current_idx)
        #     num_samples_per_class = private_labels_counter[class_]
        #     if num_samples_per_class == 0:
        #         continue
        #     syn_samples_per_class = rephrase_data[current_idx: current_idx + num_samples_per_class]
        #     syn_labels_per_class = all_private_labels[current_idx: current_idx + num_samples_per_class]
        #     new_variants_samples_stacked, _ = api.text_variation(syn_samples_per_class, syn_labels_per_class, args.combine_divide-1, args.variation_type,)
        #     syn_samples.extend(syn_samples_per_class)  # seed samples
        #     for x in new_variants_samples_stacked:  # L-1 variations
        #         syn_samples.extend(x.tolist())
        #     syn_labels.extend(
        #         syn_labels_per_class * args.combine_divide)
        #     current_idx += num_samples_per_class
        #     sync_labels_counter[class_] = num_samples_per_class * args.combine_divide
        syn_samples, syn_labels = api.text_variation(rephrase_data, all_private_labels, num_variations_per_sample=args.combine_divide-1, variation_type=args.variation_type)
        os.makedirs(init_folder)
        save_samples(syn_samples, syn_labels, init_folder, args.dataset)
    else:
        syn_samples, syn_labels = load_samples(init_folder, args.dataset)

    sync_labels_counter = collections.Counter(syn_labels)

    args.result_folder = os.path.join(args.result_folder, str(args.noise_multiplier))
    for t in range(0, args.epochs):
        if os.path.exists(os.path.join(args.result_folder, str(t))):
            print(f'noise multiplier {args.noise_multiplier} epoch {t} already exists')
            continue
        print(f'noise multiplier {args.noise_multiplier} epoch {t}')
        logging.info(f'epoch {t}')
        packed_samples = np.expand_dims(syn_samples, axis=1)
        packed_features = []
        logging.info('Running feature extraction')
        # iterate over # lookahead_degree variations.
        for i in range(packed_samples.shape[1]):
            sub_packed_features = extract_features(
                data=packed_samples[:, i],
                batch_size=args.feature_extractor_batch_size,
                model_name=args.feature_extractor,
            )
            packed_features.append(sub_packed_features)

        # take the averaged embedding for each sequence..
        packed_features = np.mean(packed_features, axis=0)

        count = []
        current_idx = 0
        # for next iteration
        new_syn_samples = []
        new_syn_labels = []

        all_selected_samples = []
        all_selected_labels = []

        for class_i, class_ in enumerate(private_classes):
            # key must have the same order as  private_classes (from private_labels_counter)
            num_samples_per_class = sync_labels_counter[class_]
            if num_samples_per_class == 0:
                continue
            # get the count for each synthetic data
            public_features = packed_features[current_idx : num_samples_per_class+current_idx]
            logging.info(
                f'{class_}, {num_samples_per_class} , features shape {public_features.shape}')
            assert num_samples_per_class == public_features.shape[0]

            selected_size = int(num_samples_per_class/args.combine_divide)
            logging.info(f'selected_size  {selected_size}')
            if selected_size == 0:
                sub_count = []
                sub_new_indices = list(
                    range(current_idx, num_samples_per_class+current_idx))
                selected_syn_samples = [syn_samples[i] for i in sub_new_indices]
                selected_labels = [syn_labels[i] for i in sub_new_indices]
            else:
                sub_count, sub_clean_count = dp_nn_histogram(
                    public_features=public_features,
                    private_features=all_private_features[private_labels_indexer[class_]],
                    noise_multiplier=args.noise_multiplier,
                    num_nearest_neighbor=args.num_nearest_neighbor,
                    mode=args.nn_mode,
                    threshold=args.count_threshold)
                assert np.sum(sub_count) >= 0
                # Generating new indices of synthetic data
                if args.select_syn_mode == 'prob':
                    candidate_indices = np.arange(
                        current_idx, num_samples_per_class + current_idx, dtype=int)
                    sampling_prob = (sub_count) / np.sum(sub_count)
                    top_1_ind = np.argpartition(sampling_prob, -1)[-1:]
                    sub_new_indices = np.random.choice(
                        candidate_indices,
                        size=selected_size,
                        p=sampling_prob)
                    # logging.info(f'sub_new_indices size  {len(sub_new_indices)}')

                elif args.select_syn_mode == 'rank':
                    sort_index = [
                        i+current_idx for i, x in sorted(enumerate(sub_count), key=lambda x: -x[1])]
                    sub_new_indices = sort_index[:selected_size]  # top votes
                else:
                    raise ValueError(
                        f'supported select_syn_mode {args.select_syn_mode}')
                # Generate new synthetic data
                selected_syn_samples = [syn_samples[i]
                                        for i in sub_new_indices]
                selected_labels = [
                    syn_labels[i] for i in sub_new_indices]
                # logging.info(f'selected_syn_samples shape {len(selected_syn_samples)} label {len(selected_labels)}')
                assert len(selected_syn_samples) == len(
                    selected_labels)

                all_selected_samples.extend(selected_syn_samples)
                all_selected_labels.extend(selected_labels)

            current_idx += public_features.shape[0]

        new_syn_samples, new_syn_labels = api.text_variation(all_selected_samples, all_selected_labels, args.combine_divide-1, args.variation_type)
        syn_samples = new_syn_samples
        syn_labels = new_syn_labels

        save_samples(all_selected_samples, all_selected_labels, f'{args.result_folder}/{t}', args.dataset)
        save_samples(syn_samples, syn_labels, f'{args.result_folder}/{t}', args.dataset, filename='new_syn_samples.json')


if __name__ == '__main__':
    main()