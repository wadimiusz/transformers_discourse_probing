import argparse
import logging
import os
import sys
import gc

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config


class DiscourseDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, pad_token_id, max_len):
        self.tokens = tokens
        self.pad_token_id = pad_token_id
        self.max_len = max_len
    
    def __getitem__(self, item):
        sentence = [int(x) for x in self.tokens[item][:self.max_len]]
        return torch.LongTensor(sentence + [self.pad_token_id] * (self.max_len - len(sentence))).cuda(),\
               torch.LongTensor([1] * len(sentence) + [0] * (self.max_len - len(sentence))).cuda()
    
    def __len__(self):
        return len(self.tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_layer_num', type=int,
                        help="Number 0..48 of the layer to get hidden states from")
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    config = GPT2Config.from_pretrained('gpt2-medium',
                                        output_hidden_states=True)
    gpt2 = GPT2Model.from_pretrained('gpt2-medium', config=config).cuda()
    logging.getLogger("transformers.tokenization_utils").setLevel(
        logging.ERROR)

    for subsample in ["train", "test"]:
        if not os.path.isdir(subsample):
            os.mkdir(subsample)

        df = pd.read_csv('{}.csv'.format(subsample))
        if os.path.isfile(f'{subsample}_tokens_gpt2.pkl'):
            print("Loading token ids...", file=sys.stderr)
            tokens = joblib.load(f'{subsample}_gpt2.pkl')
        else:
            print("Transforming texts to token ids...", file=sys.stderr)
            tokens = [tokenizer.encode(x) for x in tqdm(df.texts)]
            joblib.dump(tokens, f'{subsample}_gpt2.pkl')
        dataset = DiscourseDataset(tokens, pad_token_id=0, max_len=config.n_positions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
        gpt2.eval()
        mean_results, max_results = list(),  list()
        with torch.no_grad():
            for num, (token_ids, attention_ids) in enumerate(tqdm(dataloader), 1):
                _, _, hidden_states = gpt2(token_ids,
                                           attention_mask=attention_ids)
                hidden_states_cpu = [x.cpu().numpy() for x in hidden_states]
                del hidden_states
                gc.collect()

                output = hidden_states_cpu[args.hidden_layer_num]
                del hidden_states_cpu

                sentence_lens = attention_ids.sum(1).cpu().numpy()

                output_zero_padding = output.transpose([2, 0, 1]) * attention_ids.cpu().numpy()
                output_zero_padding = output_zero_padding.transpose([1, 2, 0])

                mean_result = (output_zero_padding.sum(1).T / sentence_lens).T
                max_result = np.array([matrix[:length].max(0) for matrix, length in zip(output_zero_padding, sentence_lens)])

                mean_results.append(mean_result)
                max_results.append(max_result)

                torch.cuda.empty_cache()

        np.save(f'{subsample}/gpt2_mean_embeddings_layer_{args.hidden_layer_num}', np.vstack(mean_results))
        np.save(f'{subsample}/gpt2_max_embeddings_layer_{args.hidden_layer_num}', np.vstack(max_results))


if __name__ == "__main__":
    main()