import gc
import json
import os
import random
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from nltk.tokenize.toktok import ToktokTokenizer
from prettytable import PrettyTable
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm


class ExperimentalSetup:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.reltypes = self.config['reltypes']
        self.groups_to_reltypes = self.config['groups_to_reltypes']
        self.df_train = pd.read_csv('train.csv')
        self.df_test = pd.read_csv('test.csv')

        self.reltypes_to_groups = {reltype: group for group in
                                   self.groups_to_reltypes for reltype in
                                   self.groups_to_reltypes[group]}

        self.internal_classification_groups_to_reltypes = None
        self.toktok = ToktokTokenizer()
        self.tfidf = TfidfVectorizer(tokenizer=self.toktok.tokenize,
                                     lowercase=True)

    def load_text_vectors(self, subset, name):
        if name == "random":
            if subset == "train":
                length = len(self.df_train)
            else:
                length = len(self.df_test)

            return np.random.random((length, 768))

        if name in ["num_bert_tokens", "num_toktok_tokens", "num_chars",
                    "num_sents"]:
            if subset == 'train':
                return np.array(self.df_train[name]).reshape(-1, 1)
            else:
                return np.array(self.df_test[name]).reshape(-1, 1)

        if name == "tfidf_bag_of_words":
            if subset == "train":
                return self.tfidf.fit_transform(self.df_train['texts'])
            else:
                return self.tfidf.transform(self.df_test['texts'])

        return np.load(f"{subset}/{name}.npy")

    def get_relation_group(self, relation):
        for group_name, relations in self.groups_to_reltypes.items():
            for another_relation in relations:
                if another_relation.lower() in relation.lower():
                    return another_relation, group_name
        raise ValueError(f"Relation {relation} does not belong to any group")

    def standardize_relation_name(self, relation):
        for reltype in self.reltypes_to_groups:
            if relation.lower() in reltype:
                return reltype

    def extract_task_targets(self, df: pd.DataFrame):
        df["tree_depth_3_classes"] = df.tree_depth.copy()
        df["tree_depth_5_classes"] = df.tree_depth.copy()

        df["tree_depth_3_classes"] = df.tree_depth_3_classes.clip(upper=3)
        df["tree_depth_5_classes"] = df.tree_depth_5_classes.clip(upper=5)

        df["internal_rel2pars"] = [eval(x) for x in df.internal_rel2pars]

        df["relation_groups"] = [set(
            [None if relation is None else self.get_relation_group(relation)[1]
             for
             relation in rels]) for rels in df.internal_rel2pars]

        for reltype in self.reltypes:
            df[reltype] = [any(
                rel.lower().startswith(reltype) for rel in x if
                rel is not None)
                for x in df.internal_rel2pars]

        for outermost_reltype in self.reltypes:
            df[f"outermost_{outermost_reltype}"] = [any(
                rel.lower().startswith(outermost_reltype) for rel in x if
                rel is not None) for x in df.internal_rel2pars]

        for group_name in self.groups_to_reltypes.keys():
            if group_name != 'structural':
                df[f"{group_name}_group"] = [group_name in x for x in
                                             df.relation_groups]

        if self.internal_classification_groups_to_reltypes is None:
            groups_to_reltypes = self.groups_to_reltypes
            internal_classification_groups_to_reltypes = dict()
        else:
            groups_to_reltypes = self.internal_classification_groups_to_reltypes

        for group_name in tqdm(groups_to_reltypes.keys()):
            result = list()
            for rel2pars in tqdm(df.internal_rel2pars):
                groups = [self.get_relation_group(x) for x in rel2pars if
                          x is not None]
                groups = [(relation, group) for relation, group in groups if
                          group is not None and group == group_name]

                if len(groups) == 1:
                    result.append(groups[0][0])
                else:
                    result.append(None)

            counter = Counter(result)
            if self.internal_classification_groups_to_reltypes is None:
                popular_classes = [elem for elem, num in counter.most_common()
                                   if num >= 100 and elem is not None]
                if len(popular_classes) < 2:
                    continue

                internal_classification_groups_to_reltypes[
                    group_name] = popular_classes
            else:
                popular_classes = \
                self.internal_classification_groups_to_reltypes[group_name]

            result = [x if x in popular_classes else None for x in result]
            df[f"{group_name}_internal_classification"] = result

        if self.internal_classification_groups_to_reltypes is None:
            self.internal_classification_groups_to_reltypes = internal_classification_groups_to_reltypes

        return df

    @staticmethod
    def get_baselines(df: pd.DataFrame):
        bert_len_baseline = np.array(df["num_bert_tokens"]).reshape((-1, 1))
        toktok_len_baseline = np.array(df["num_toktok_tokens"]).reshape(
            (-1, 1))
        char_len_baseline = np.array(df["num_chars"]).reshape((-1, 1))
        sent_len_baseline = np.array(df["num_sents"]).reshape((-1, 1))
        return [bert_len_baseline, toktok_len_baseline,
                char_len_baseline, sent_len_baseline]

    def get_masks_and_names(self):
        def bert_len_mask(df): return df.num_bert_tokens < 512

        def cap_mask(df): return df.texts.str[0].str.isupper()

        def period_mask(df): return df.texts.str.endswith('.')

        def question_mask(df): return df.texts.str.endswith('?')

        def exclamation_mask(df): return df.texts.str.endswith('!')

        def punct_mask(df): return period_mask(df) | question_mask(
            df) | exclamation_mask(df)

        def full_sentence_mask(df): return bert_len_mask(df) & cap_mask(
            df) & punct_mask(df)

        def one_sentence_mask(df): return full_sentence_mask(df) & (
                    df.num_sents == 1)

        def fifteen_tokens_mask(df): return full_sentence_mask(df) & (
                    df.num_toktok_tokens == 15)

        def fifteen_twenty_five_tokens_mask(df): return full_sentence_mask(
            df) & (df.num_toktok_tokens >= 15) & (df.num_toktok_tokens <= 20)

        def one_sentence_tokens_15_20_mask(df): return one_sentence_mask(
            df) & (df.num_toktok_tokens >= 15) & (df.num_toktok_tokens <= 20)

        def one_sentence_tokens_15_25_mask(df):
            return one_sentence_mask(df) & (df.num_toktok_tokens >= 15) & (
                    df.num_toktok_tokens <= 25)

        masks = [bert_len_mask, full_sentence_mask, one_sentence_mask,
                 fifteen_tokens_mask,
                 fifteen_twenty_five_tokens_mask,
                 one_sentence_tokens_15_20_mask,
                 one_sentence_tokens_15_25_mask]

        mask_names = ["bert_len_mask", "full_sentence", "one_sentence",
                      "fifteen_tokens",
                      "fifteen_twenty_five_tokens",
                      "one_sentence_15_20_tokens", "one_sentence_15_25_tokens"]

        for group_name, classes in self.internal_classification_groups_to_reltypes.items():
            def mask(df, group_name=group_name, classes=classes):
                return pd.Series([x in classes for x in
                                  df[f"{group_name}_internal_classification"]])

            mask_name = f"{group_name}_internal_classification_mask"

            masks.append(mask)
            mask_names.append(mask_name)
        return mask_names, masks

    def perform_experiments(self):
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')

        embedding_names = [f"{pooling}_embeddings_layer_{i}" for pooling in
                           ["cls", "mean", "max", 'sep'] for i in range(13)]
        embedding_names.extend(
            [f"gpt2_{pooling}_embeddings_layer_{i}" for pooling in
             ["mean", "max"] for i in [4, 8, 12, 16, 20, 24]])
        embedding_names.extend(
            ["w2v_mean_embeddings", "w2v_tfidf_mean_embeddings",
             "tfidf_bag_of_words",
             "num_bert_tokens",
             "num_toktok_tokens",
             "num_chars",
             "num_sents",
             "random"])

        for name in embedding_names:
            if name not in ["num_bert_tokens", "num_toktok_tokens",
                            "num_chars", "num_sents", 'random',
                            'tfidf_bag_of_words']:
                assert os.path.isfile(os.path.join("train",
                                                   f"{name}.npy")), f"train/{name} not found"
                assert os.path.isfile(os.path.join("test",
                                                   f"{name}.npy")), f"test/{name} not found"

        df_train = self.extract_task_targets(df_train)
        df_test = self.extract_task_targets(df_test)

        mask_names, masks = self.get_masks_and_names()
        names_to_masks = {name: mask for name, mask in zip(mask_names, masks)}

        group_tasks = [f"{x}_group" for x in self.groups_to_reltypes.keys() if
                       x != 'structural']
        internal_classification_tasks = [f"{x}_internal_classification" for x
                                         in
                                         self.internal_classification_groups_to_reltypes.keys()]

        tasks = ["is_leaf", "tree_depth_3_classes",
                 "tree_depth_5_classes"] + self.reltypes + group_tasks + internal_classification_tasks
        masks_for_tasks = [["full_sentence"]] * (
                    3 + len(self.reltypes) + len(group_tasks)) + [
                              [f"{x}_internal_classification_mask"] for x in
                              self.internal_classification_groups_to_reltypes.keys()]

        assert len(tasks) == len(
            masks_for_tasks), f"{len(tasks)} tasks != {len(masks_for_tasks)} masks_for_tasks"
        print(f"Full list of tasks: {tasks}\n", file=sys.stderr)
        print(f"Full list of embedding names: {embedding_names}\n",
              file=sys.stderr)
        for task, mask_for_task in tqdm(zip(tasks, masks_for_tasks),
                                        total=len(tasks)):

            print(f"\nNow doing task `{task}`", file=sys.stderr)
            results = np.zeros((len(mask_for_task), len(embedding_names),))
            results[:] = np.nan

            train_sample_size_table = PrettyTable()
            test_sample_size_table = PrettyTable()

            classes = sorted(set(df_train[task]) & set(df_test[task]) - {None})
            train_sample_size_table.add_column("", [f"`{x}`" for x in classes])
            test_sample_size_table.add_column("", [f"`{x}`" for x in classes])

            assert len(mask_for_task) == 1

            for i, mask_name in enumerate(tqdm(mask_for_task)):
                print(f"\nNow doing mask `{mask_name}`", file=sys.stderr)
                mask = names_to_masks[mask_name]
                train_mask = np.array(mask(df_train))
                test_mask = np.array(mask(df_test))

                df_train_masked, df_test_masked = df_train[train_mask], \
                                                  df_test[test_mask]
                train_sample_size_table.add_column(f"`{mask_name}`", [
                    (df_train_masked[task] == cls).sum() for cls in classes])
                test_sample_size_table.add_column(f"`{mask_name}`", [
                    (df_test_masked[task] == cls).sum() for cls in classes])

                for j, embedding_name in enumerate(tqdm(embedding_names)):
                    print(
                        f"\nNow trying vectorization method `{embedding_name}`",
                        file=sys.stderr)
                    train_matrix = self.load_text_vectors("train",
                                                          embedding_name)
                    test_matrix = self.load_text_vectors("test",
                                                         embedding_name)

                    assert train_matrix.shape[0] == len(self.df_train)
                    assert test_matrix.shape[0] == len(self.df_test)

                    X_train = train_matrix[train_mask]
                    y_train = np.array(df_train_masked[task])

                    X_test = test_matrix[test_mask]
                    y_test = np.array(df_test_masked[task])

                    assert X_train.shape[0] == y_train.shape[0]
                    assert X_test.shape[0] == y_test.shape[0]

                    classes = sorted(set(y_train))
                    class_weight = compute_class_weight('balanced', classes,
                                                        y_train)
                    class_weight = {cls: weight for cls, weight in
                                    zip(classes, class_weight)}
                    scores = list()

                    if len(np.unique(y_test)) < 2 or len(
                            np.unique(y_test)) < 2:
                        mean_score = np.nan
                    else:
                        for _ in tqdm(range(5)):
                            indices = list(range(X_train.shape[0]))
                            random.shuffle(indices)
                            clf = LogisticRegression(class_weight=class_weight,
                                                     n_jobs=1, verbose=0).fit(
                                X_train[indices], y_train[indices])

                            proba = clf.predict_proba(X_test)

                            if len(set(y_train)) <= 2:
                                score = roc_auc_score(y_test, proba[:, 1])

                            else:
                                score = roc_auc_score(y_test, proba,
                                                      multi_class='ovr')

                            assert len(proba) == len(y_test) == len(
                                list(df_test_masked.texts))
                            scores.append(score)

                        mean_score = np.mean(scores)

                    results[i, j] = mean_score
                    del train_matrix
                    del test_matrix
                    unreachable_items = gc.collect()
                    print(f"{unreachable_items} unreachable items deleted",
                          file=sys.stderr)

            scores_table = PrettyTable()
            scores_table.add_column("", [f"`{x}`" for x in embedding_names])
            for mask_name, results_row in zip(mask_for_task, results):
                values = [f"{x:.3f}" for x in list(results_row)]
                if not np.isnan(results_row).all():
                    values[
                        results_row.argmax()] = f"**{values[results_row.argmax()]}**"
                scores_table.add_column(f"`{mask_name}`", values)
            print(f"RESULTS FOR TASK `{task}`:\n```\n{scores_table}\n```\n")
            print(
                f"SAMPLE SIZES IN TRAINING SET:\n```\n{train_sample_size_table}\n```\n")
            print(
                f"SAMPLE SIZES IN TEST SET:\n```\n{test_sample_size_table}\n```\n")


def main():
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    setup = ExperimentalSetup()
    setup.perform_experiments()


if __name__ == "__main__":
    main()
