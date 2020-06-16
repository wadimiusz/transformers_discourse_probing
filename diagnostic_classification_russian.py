import gc
import json
import sys
import warnings
from collections import Counter

import gensim
import numpy as np
import pandas as pd
import prettytable
from nltk.tokenize import sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import BertTokenizer
from collections import defaultdict

class ExperimentSetup:
    def __init__(self, config_path='config_russian.json', df_path='russian.csv'):
        with open(config_path) as f:
            self.config = json.load(f)

        self.groups = self.config['groups']
        self.relations = self.config['relations']
        self.masks = self.config['masks']
        self.groups_classification = self.config['groups_classification']
        self.df = pd.read_csv(df_path)
        self.df["relations"] = [eval(x) for x in self.df["relations"]]
        self.toktok = ToktokTokenizer()
        self.fasttext = gensim.models.KeyedVectors.load(self.config['fasttext_path'])
        self.text_vectors = self.get_text_vectors(self.df.text)
        self.berttokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

    def get_text_vectors(self, texts):
        matrix_list = list()
        for text in texts:
            matrix = list()
            for token in self.toktok.tokenize(text.lower()):
                if token in self.fasttext:
                    matrix.append(self.fasttext[token])

            if len(matrix) == 0:
                matrix = np.zeros((1, self.fasttext.vectors.shape[1]))
            else:
                matrix = np.array(matrix)

            matrix_list.append(matrix)

        return matrix_list

    def extract_targets(self):
        df_targets = self.df.copy(deep=True)
        df_targets['depth_2_classes'] = df_targets['depth'].clip(upper=2)
        df_targets["depth_5_classes"] = df_targets["depth"].clip(upper=5)
        df_targets["depth_10_classes"] = df_targets["depth"].clip(upper=10)

        for relation in self.relations:
            df_targets[f"{relation}_binary"] = [(relation in relations) for relations in df_targets["relations"]]

        for relation_type, relations in self.groups_classification.items():
            internal = list()
            for relations_in_text in df_targets.relations:
                relations_set = set(x for x in relations_in_text if x in relations)
                if len(relations_set) == 1:
                    internal.append(list(relations_set)[0])
                else:
                    internal.append(None)

            df_targets[f'{relation_type}_internal'] = internal
            df_targets[f'{relation_type}_group_binary'] = [any(y in x for y in relations) for x in df_targets.relations]

        return df_targets

    def extract_tfidf(self, df_train, df_test):
        tfidf = TfidfVectorizer(tokenizer=self.toktok.tokenize, lowercase=True)
        X_train = tfidf.fit_transform(df_train.text)
        X_test = tfidf.transform(df_test.text)
        return X_train, X_test

    def get_document_frequency(self, texts):
        doc_freq = Counter()
        for text in texts:
            for token in set(self.toktok.tokenize(text.lower())):
                doc_freq[token] += 1

        return doc_freq

    def get_idf_lists(self, texts, doc_freq, num_documents):
        idf_lists = list()
        for text in texts:
            idf_list = list()
            for token in self.toktok.tokenize(text):
                if token in self.fasttext:
                    if token in doc_freq:
                        idf_list.append(np.log(num_documents / doc_freq[token]))
                    else:
                        idf_list.append(0.)

            idf_lists.append(np.array(idf_list))

        return idf_lists

    def get_fasttext_tfidf_matrix(self, df_train, df_test):
        doc_freq = self.get_document_frequency(df_train.text)

        train_word_vectors = [self.text_vectors[idx] for idx in df_train.index]
        test_word_vectors = [self.text_vectors[idx] for idx in df_test.index]

        train_idf_lists = self.get_idf_lists(df_train.text, doc_freq, len(df_train))
        test_idf_lists = self.get_idf_lists(df_test.text, doc_freq,len(df_test))

        assert len(train_word_vectors) == len(train_idf_lists)
        assert len(test_word_vectors) == len(test_idf_lists)

        for vectors, idfs in zip(train_word_vectors, train_idf_lists):
            assert len(vectors) == len(idfs)

        for vectors, idfs in zip(test_word_vectors, test_idf_lists):
            assert len(vectors) == len(idfs)

        train_matrix = [(np.array(word_vectors).T * idf_vector).T.mean(0) for word_vectors, idf_vector in zip(train_word_vectors, train_idf_lists)]
        test_matrix = [(np.array(word_vectors).T * idf_vector).T.mean(0) for word_vectors, idf_vector in zip(test_word_vectors, test_idf_lists)]

        return train_matrix, test_matrix

    def get_matrix(self, name, df_train, df_test):
        if name == "random":
            train_len = len(df_train)
            test_len = len(df_test)
            return np.random.random((train_len, 768)), np.random.random((test_len, 768))
        elif name == "tfidf_bag_of_words":
            return self.extract_tfidf(df_train, df_test)
        elif name == "fasttext_mean_embeddings":
            train_matrix = np.array([np.mean(self.text_vectors[idx], 0) for idx in df_train.index])
            test_matrix = np.array([np.mean(self.text_vectors[idx], 0) for idx in df_test.index])

            assert train_matrix.shape == (len(df_train), self.fasttext.vectors.shape[1])
            assert test_matrix.shape == (len(df_test), self.fasttext.vectors.shape[1])

            return train_matrix, test_matrix
        elif name == "fasttext_tfidf_mean_embeddings":
            return self.get_fasttext_tfidf_matrix(df_train, df_test)
        elif name == "num_toktok_tokens":
            train_toktok_tokens = [[len(self.toktok.tokenize(x))] for x in tqdm(df_train.text)]
            test_toktok_tokens = [[len(self.toktok.tokenize(x))] for x in tqdm(df_test.text)]
            return train_toktok_tokens, test_toktok_tokens
        elif name == "num_chars":
            return [[len(x)] for x in df_train.text], [[len(x)] for x in df_test.text]
        elif name == "num_sents":
            return [[len(sent_tokenize(x))] for x in df_train.text], [[len(sent_tokenize(x))] for x in df_test.text]
        else:
            train_idx = df_train.index
            test_idx = df_test.index
            matrix = np.load(f'russian/{name}.npy')
            return matrix[train_idx], matrix[test_idx]

    def get_masks(self):
        def all_mask(df): return pd.Series([True] * len(df))

        def bert_len(df): return df.num_bert_tokens < 512
        def cap(df): return df.text.str[0].str.isupper()

        def period(df): return df.text.str.endswith('.')
        def question(df): return df.text.str.endswith('?')
        def exclamation(df): return df.text.str.endswith('!')

        def punct_mask(df): return period(df) | question(df) | exclamation(df)

        def full_sentence(df): return bert_len(df) & cap(df) & punct_mask(df)
        def one_sentence(df): return full_sentence(df) & (df.num_sents == 1)

        result = {"all": all_mask, "bert_len": bert_len,
                  "full_sentence": full_sentence, "one_sentence": one_sentence}

        for relation_type, relations in self.groups_classification.items():
            def masking_function(df, relations=relations):
                return pd.Series([sum(y in relations for y in x) == 1 for x in df.relations])

            result[f"{relation_type}_internal"] = masking_function

        return result

    def perform_experiments(self):
        kfold = KFold(n_splits=5, shuffle=True)
        file_ids = np.arange(178)
        df = self.extract_targets()

        embedding_names = ['random', 'tfidf_bag_of_words', 'fasttext_mean_embeddings', 'fasttext_tfidf_mean_embeddings',
                           "num_toktok_tokens", "num_chars", "num_sents", "num_bert_tokens"]
        embedding_names.extend([f"{pooling}_embeddings_layer_{i}" for pooling in ["cls", "mean", "max", "sep"] for i in range(13)])
        masks = self.get_masks()
        targets = list()
        targets += ['depth_2_classes', "depth_5_classes", "depth_10_classes"]
        targets += [f"{relation}_binary" for relation in self.relations]
        targets += [f"{relation_type}_internal" for relation_type in self.groups_classification.keys()]
        targets += [f"{relation_type}_group_binary" for relation_type in self.groups_classification.keys()]
        for target in targets:
            assert target in self.masks, f"No masks found for target {target}"
            assert target in df.columns, f"Target {target} not found in dataframe"

        for task_name in tqdm(targets):
            print(f"\n\nNow doing {task_name}...\n\n", file=sys.stderr)
            score_table = prettytable.PrettyTable()
            score_table.add_column("", [f"{x}" for x in embedding_names])
            assert len(self.masks[task_name]) == 1
            for mask_name in tqdm(self.masks[task_name]):
                masking_function = masks[mask_name]

                score_by_embedding = list()
                mask = masking_function(self.df)
                for matrix_name in tqdm(embedding_names):

                    y = np.array(df[task_name])
                    scores = list()
                    for train_index, test_index in tqdm(kfold.split(file_ids), total=5):
                        if np.isnan(scores).any():
                            print(f"Found nan in scores. Trying to continue...", file=sys.stderr)
                            scores.append(np.nan)

                        train_mask = np.array(df.file_id.isin(train_index)) & mask
                        test_mask = np.array(df.file_id.isin(test_index)) & mask

                        df_train = df[train_mask]
                        df_test = df[test_mask]

                        X_train, X_test = self.get_matrix(matrix_name, df_train, df_test)

                        y_train = y[train_mask]
                        y_test = y[test_mask]

                        if len(set(y_train)) < 2 or len(set(y_train)) != len(set(y_test)):
                            print(f"set of train labels: {set(y_train)}, set of test labels: {set(y_test)}, trying to continue", file=sys.stderr)
                            scores.append(np.nan)
                            continue

                        classes = sorted(set(y_train))
                        class_weight = compute_class_weight('balanced', classes, y_train)
                        class_weight = {cls: weight for cls, weight in zip(classes, class_weight)}

                        clf = LogisticRegression(class_weight=class_weight).fit(X_train, y_train)
                        proba = clf.predict_proba(X_test)
                        if len(classes) == 2:
                            score = roc_auc_score(y_test, proba[:, 1])
                        else:
                            score = roc_auc_score(y_test, proba, multi_class='ovr')

                        assert len(proba) == len(list(df_test.text)) == len(y_test)
                        scores.append(score)

                        del X_train
                        del X_test
                        gc.collect()

                    mean_score = np.mean(scores)
                    score_by_embedding.append(mean_score)

                max_score = np.max(score_by_embedding)
                score_by_embedding = [(f"**{x:.3f}**" if x == max_score else f"{x:.3f}") for x in score_by_embedding]
                score_table.add_column(mask_name, score_by_embedding)

            print(f"RESULTS FOR TASK {task_name}:\n```\n{score_table}```")


def main():
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    setup = ExperimentSetup()
    setup.perform_experiments()


if __name__ == "__main__":
    main()
