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
from transformers import BertTokenizer, BertModel, BertConfig


class DiscourseDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, pad_token_id):
        self.tokens = tokens
        self.pad_token_id = pad_token_id
    
    def __getitem__(self, item):
        sentence = self.tokens[item][:512]
        return torch.LongTensor(sentence + [self.pad_token_id] * (512 - len(sentence))).cuda(),\
               torch.LongTensor([1] * len(sentence) + [0] * (512 - len(sentence))).cuda()
    
    def __len__(self):
        return len(self.tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_layer_num', type=int,
                        help="Number 1..12 of the layer to get hidden states from")
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased',
                                        output_hidden_states=True)
    bert = BertModel.from_pretrained('bert-base-uncased', config=config).cuda()
    logging.getLogger("transformers.tokenization_utils").setLevel(
        logging.ERROR)

    for subsample in ["train", 'test']:
        if not os.path.isdir(subsample):
            os.mkdir(subsample)

        df = pd.read_csv('{}.csv'.format(subsample))
        if os.path.isfile(f'{subsample}_tokens.pkl'):
            print("Loading token ids...", file=sys.stderr)
            tokens = joblib.load(f'{subsample}_tokens.pkl')
        else:
            print("Transforming texts to token ids...", file=sys.stderr)
            tokens = [tokenizer.encode(x) for x in tqdm(df.texts)]
            joblib.dump(tokens, f'{subsample}_tokens.pkl')
        dataset = DiscourseDataset(tokens, tokenizer.pad_token_id)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
        bert.eval()
        mean_results, max_results, cls_results, sep_results = list(), list(), list(), list()
        with torch.no_grad():
            for token_ids, attention_ids in tqdm(dataloader):
                gc.collect()
                _, _, hidden_states = bert(token_ids, attention_ids)

                attention_ids = attention_ids.cpu()

                output = hidden_states[args.hidden_layer_num].cpu()  # 0..12
                del hidden_states
                gc.collect()

                sentence_lens = attention_ids.sum(1)

                output_zero_padding = output.permute([2, 0, 1]) * attention_ids
                del attention_ids
                output_zero_padding = output_zero_padding.permute([1, 2, 0])

                mean_result = (output_zero_padding.sum(1).T / sentence_lens).T
                # max_result = output_neginf_padding.max(1).values
                assert len(output_zero_padding) == len(sentence_lens)
                max_result = [matrix[:length].cpu().numpy().max(0) for matrix, length in zip(output_zero_padding, sentence_lens)]
                cls_result = output[:, 0]
                del output

                mean_results.append(mean_result.cpu().numpy())
                max_results.extend(max_result)
                cls_results.append(cls_result.cpu().numpy())
                sep_results.extend(output_zero_padding[i,length-1,:].cpu().numpy() for i, length in enumerate(sentence_lens))

                del mean_result
                del max_result
                del cls_result
                gc.collect()

        np.save(f'{subsample}/max_embeddings_layer_{args.hidden_layer_num}', np.vstack(max_results))
        np.save(f'{subsample}/mean_embeddings_layer_{args.hidden_layer_num}', np.vstack(mean_results))
        np.save(f'{subsample}/cls_embeddings_layer_{args.hidden_layer_num}', np.vstack(cls_results))
        np.save(f'{subsample}/sep_embeddings_layer_{args.hidden_layer_num}', np.vstack(sep_results))


if __name__ == "__main__":
    main()