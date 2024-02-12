import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.special import softmax
from typing import List, Tuple, Dict
from utils import *
import wandb

torch.cuda.empty_cache()

class SentenceDataset(Dataset):
    """
    A dataset class for sentences, designed to be used with PyTorch's Dataset interface.

    :param sentences: A list of sentences to be encoded.
    :param tokenizer: The tokenizer used to encode the sentences.
    :param max_length: The maximum length of the encoded sentences.
    :param relations: Optional; A list of relations corresponding to each sentence. Default is None.
    :param labels: Optional; A list of labels corresponding to each sentence. Default is None.
    """

    def __init__(self, sentences, tokenizer, max_length, relations=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.relations = relations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the number of sentences in the dataset.

        :return: The number of sentences.
        """
        
        return len(self.sentences)

    def encode(self, sentence):
        raise NotImplementedError("This method should be overridden in subclass")

    def __getitem__(self, idx):
        """
        Returns an encoded item at a specified index.

        :param idx: The index of the item.
        :return: A dictionary containing encoded sentence with optional labels and relations.
        """

        sentence = self.sentences[idx]
        encoding = self.encode(sentence)

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }
        
        if self.labels is not None:
            target = self.labels[idx]
            item['labels'] = torch.tensor(target, dtype=torch.long)

        if self.relations is not None:
            relation = self.relations[idx]
            item['relations'] = torch.tensor(relation, dtype=torch.long)

        return item


class SentencePairDataset(SentenceDataset):
    """
    A dataset class for sentence pairs (head, tail) used in relation prediction, inheriting from SentenceDataset.
    """

    def encode(self, sentence_pair):
        """
        Encodes a pair of sentences into a single input sequence for the model.

        :param sentence_pair: A tuple containing the head and tail sentences.
        :return: A dictionary containing encoded sentence pair with attention mask, token type ids, etc.
        """

        head, tail = sentence_pair
        formatted_sentence = f"{head} [SEP] {tail}"
        return self.tokenizer.encode_plus(
            formatted_sentence, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )


class SentenceTripleDataset(SentenceDataset):
    """
    A dataset class for sentence triples used in triple classification, inheriting from SentenceDataset.
    """

    def encode(self, sentence_triple):
        """
        Encodes a triple of sentences (head, relation, tail) into a single input sequence for the model.

        :param sentence_triple: A tuple containing the head, relation, and tail sentences.
        :return: A dictionary containing encoded sentence triple with attention mask, token type ids, etc.
        """

        head, relation, tail = sentence_triple
        formatted_sentence = f"{head} [SEP] {relation} [SEP] {tail}"
        return self.tokenizer.encode_plus(
            formatted_sentence,
            add_special_tokens=True, 
            max_length=self.max_length, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )


class BertClassifier():
    """
    A classifier based on the BERT model architecture for sequence classification tasks.

    :param model_variant: Specifies the BERT model variant to use.
    :param num_labels: The number of labels for the classification task.
    :param parallelize: Indicates whether to use data parallelism ("T" for true, "F" for false).
    :param model_to_be_loaded: Optional; specifies the model weights to be loaded.
    :param classifier_variant: Specifies the variant of the classifier to be used, default is "default".
    """

    def __init__(self, model_variant, num_labels, parallelize="F", model_to_be_loaded=None,
                 classifier_variant="default"):
        self.initialize_model(model_variant, num_labels, classifier_variant)
        self.initialize_device()
        if model_to_be_loaded:
            self.load_model_weights(model_to_be_loaded)
        self.set_parallelization(parallelize)
        self.initialize_training_parameters()

    def initialize_model(self, model_variant, num_labels, classifier_variant):
        """
        Initializes the model, tokenizer, and sets the model variant and classifier variant.

        :param model_variant: The specific model variant to use.
        :param num_labels: The number of labels for the classification task.
        :param classifier_variant: The classifier variant to use.
        """

        self.model_variant = model_variant
        self.classifier_variant = classifier_variant
        self.num_labels = num_labels
        self.safe_model_variant = model_variant.replace("/", "-")

        tokenizer_options = {
            "distilbert" : lambda: DistilBertTokenizer.from_pretrained(
                model_variant),
            "taser": lambda: AutoTokenizer.from_pretrained(
                "dwsunimannheim/TaSeR"),
        }
        self.tokenizer = tokenizer_options[classifier_variant]()

        model_options = {
            "distilbert": lambda: DistilBertForSequenceClassification.from_pretrained(
                model_variant, num_labels=num_labels),
            "taser": lambda: AutoModelForSequenceClassification.from_pretrained(
                "dwsunimannheim/TaSeR"),
        }
        self.model = model_options[classifier_variant]()

    def initialize_device(self):
        """
        Initializes the device (CPU or CUDA) for the model.
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def load_model_weights(self, model_to_be_loaded):
        """
        Loads model weights from a specified file.

        :param model_to_be_loaded: The name of the file containing the model weights to be loaded.
        """

        weights_filename = f"../models/{self.safe_model_variant}/{model_to_be_loaded}_weights.pth"
        if os.path.exists(weights_filename):
            state_dict = torch.load(weights_filename)
            is_parallel = all([k.startswith('module.') for k in state_dict.keys()])
            if is_parallel:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            sys.exit(f"No saved model weights found at {weights_filename}")

    def set_parallelization(self, parallelize):
        """
        Sets the model parallelization.

        :param parallelize: Indicates whether to parallelize the model processing ("T" for true, "F" for false).
        """

        if parallelize == "T":
            self.model = nn.DataParallel(self.model)

    def initialize_training_parameters(self):
        """
        Initiallizes the parameter for training.
        """

        self.lr = None
        self.epochs = None
        self.batch_size = None
        self.optimizer = None
        self.class_weights = None

    def custom_weighted_cross_entropy(self, logits, labels, relationship_labels):
        """
        Computes a custom weighted cross-entropy loss for a batch of predictions.

        :param logits: The logits predicted by the model.
        :param labels: The ground truth labels.
        :param relationship_labels: Additional relationship labels for weighting the loss.
        :return: The mean weighted cross-entropy loss for the batch.
        """

        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        class_weights = torch.tensor(self.class_weights).to(self.device)
        sample_weights = class_weights[labels, relationship_labels]
        
        weighted_loss = ce_loss * sample_weights
        
        return weighted_loss.mean()

    def _train_epoch(self, data_loader):
        """
        Trains the model for one epoch.

        :param data_loader: The DataLoader providing the training batches.
        :return: The average loss over the training epoch.
        """

        total_loss = 0
        total_steps = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            # because the relation&class weights are only for TC, exclude this for RP
            if self.task_type == "TC":
                relations = batch['relations'].to(self.device)

            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels}
            
            # this is also because of the the relation&class weights that are only for TC
            if self.task_type == "TC":
                batch['relations'] = relations

            self.optimizer.zero_grad()

            expected_args = self.model.forward.__code__.co_varnames
            filtered_batch = {k: v for k, v in batch.items() if k in expected_args}
            outputs = self.model(**filtered_batch)
            
            if is_nested(self.class_weights) is False or self.class_weights is None:
                loss_obj = CrossEntropyLoss()
                loss = loss_obj(outputs.logits, batch['labels'])
            elif is_nested(self.class_weights) is True:
                loss = self.custom_weighted_cross_entropy(outputs.logits, 
                                                          batch['labels'], batch['relations'])
            else:
                sys.exit("something is wrong with the weights")

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            wandb.log({"step_loss": loss.item()})

        return total_loss / total_steps

    def train(self, data, labels, lr, epochs, batch_size, relations, class_weights=None):
        """
        Trains the model.

        :param data: The training data.
        :param labels: The training labels.
        :param lr: Learning rate for the optimizer.
        :param epochs: Number of epochs to train for.
        :param batch_size: The size of training batches.
        :param relations: Additional relationship data for training.
        :param class_weights: Optional; class weights for weighted loss calculation.
        """

        self.model.train()
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.class_weights = class_weights

        data_loader = self.create_train_data_loader(data, labels, shuffle=True, relations=relations)

        print(f"Model uses class weights {class_weights}")

        for epoch in range(self.epochs):
            epoch_loss = self._train_epoch(data_loader)

            wandb.log({"epoch_loss": epoch_loss})


class BertClassifier_relation_prediction(BertClassifier):
    """
    A subclass for relation prediction tasks using a BERT-based classifier.

    Inherits from BertClassifier and sets task type to "RP" (Relation Prediction).
    """

    def __init__(self, model_variant, num_labels, parallelize = "F",
                 model_to_be_loaded = None, classifier_variant='default'):
        super().__init__(model_variant, num_labels, parallelize=parallelize, 
                         model_to_be_loaded = model_to_be_loaded,
                         classifier_variant=classifier_variant)
        self.task_type = "RP"

    def create_train_data_loader(self, sentence_pairs: List[Tuple[str, str]], labels: List[int], 
                                 shuffle, relations):
        """
        Creates a DataLoader for training on sentence pairs.

        :param sentence_pairs: A list of sentence pairs.
        :param labels: A list of labels for each sentence pair.
        :param shuffle: Whether to shuffle the data.
        :param relations: Relationship labels for each sentence pair.
        :return: A DataLoader for the training data.
        """

        dataset = SentencePairDataset(sentence_pairs, self.tokenizer,
                                      BERT_MAX_SEQUENCE_LENGTH, labels=labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader
    
    def create_test_data_loader(self, sentence_pairs: List[Tuple[str, str]], shuffle):
        """
        Creates a DataLoader for testing on sentence pairs.

        :param sentence_pairs: A list of sentence pairs for testing.
        :param shuffle: Whether to shuffle the test data.
        :return: A DataLoader for the test data.
        """
            
        dataset = SentencePairDataset(sentence_pairs, self.tokenizer,
                                      max_length=BERT_MAX_SEQUENCE_LENGTH)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def predict(self, sentence_pairs, label_dict: Dict[int, str],
                name_test_set, model_name, batch_size) -> float:
        """
        Makes predictions on a set of sentence pairs.

        :param sentence_pairs: A list of sentence pairs to predict on.
        :param label_dict: A dictionary mapping label indices to string labels.
        :param name_test_set: The name of the test set.
        :param model_name: The name of the model.
        :param batch_size: The batch size for prediction.
        :return: A list of predictions.
        """
            
        self.model.eval()

        self.batch_size = batch_size

        data_loader = self.create_test_data_loader(sentence_pairs, shuffle=False)

        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                _, predictions = torch.max(outputs.logits.data, 1)
                all_predictions.extend(predictions.tolist())

        if len(sentence_pairs[0]) == 3:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "relation", "tail"])
        elif len(sentence_pairs[0]) ==2:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "tail"])
        else:
            sys.exit('something went wrong in RP predict')

        df_predictions['prediction'] = all_predictions
        df_predictions = df_predictions.replace(label_dict)
        save_processed_df_to_csv(df_predictions, 
                                 f"{self.task_type}_{model_name}_{name_test_set}_prediction",
                                 "predictions")

        return all_predictions

    def evaluate(self,
                 sentence_pairs,
                 labels,
                 label_dict: Dict[int, str], 
                 name_test_set,
                 model_name,
                 batch_size) -> float:
        """
        Evaluates the model on a set of sentence pairs and labels.

        :param sentence_pairs: A list of sentence pairs.
        :param labels: The true labels for each sentence pair.
        :param label_dict: A dictionary mapping label indices to string labels.
        :param name_test_set: The name of the test set.
        :param model_name: The name of the model.
        :param batch_size: The batch size for evaluation.
        :return: The accuracy of the predictions.
        """
            
        self.model.eval()

        predictions = self.predict(sentence_pairs, 
                                   label_dict, 
                                   name_test_set, 
                                   model_name,
                                   batch_size)

        accuracy = accuracy_score(labels, predictions)
        print('Overall Accuracy: %d %%' % (100 * accuracy))
        all_label_indices = list(range(self.num_labels))

        target_names: List[str] = list(label_dict.values())
        cm: np.ndarray = confusion_matrix(y_true=labels, 
                                          y_pred=predictions, 
                                          labels = all_label_indices) 

        if len(target_names) != len(np.unique(labels)):
            print("Error is probably caused by fact that one of the classes is not in the test set")

        cm_df: pd.DataFrame = pd.DataFrame(cm, index=[f'Actual {label}' for label in target_names], 
                                       columns=[f'Predicted {label}' for label in target_names])
        print("Confusion Matrix:")
        print(cm_df)

        print("Classification Report:")

        print(classification_report(labels, predictions, 
                                    target_names=target_names, 
                                    labels=all_label_indices))

        if len(sentence_pairs[0]) == 3:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "relation", "tail"])
        elif len(sentence_pairs[0]) ==2:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "tail"])
        else:
            sys.exit('something went wrong in RP predict')

        df_predictions['relation'] = labels
        df_predictions['prediction'] = predictions
        df_predictions = df_predictions.replace(label_dict)

        res_to_save = ['Overall Accuracy: %d %%' % (100 * accuracy), "Confusion Matrix:",
                       cm_df, "Classification Report:", classification_report(
                           labels, 
                           predictions, 
                           target_names=target_names, 
                           labels=all_label_indices)]
        save_printed_output_to_file(res_to_save, f"RP_{model_name}_{name_test_set}", 
                                    "evaluation_results")
        print("saved here:", f"RP_{model_name}_{name_test_set}")

        return accuracy

class BertClassifier_triple_classification(BertClassifier):
    """
    A subclass for triple classification tasks using a BERT-based classifier.

    Inherits from BertClassifier and sets task type to "TC" (Triple Classification).
    """

    def __init__(self, model_variant, num_labels, parallelize = "F",
                 model_to_be_loaded = None, classifier_variant='default'):
        super().__init__(model_variant, num_labels, parallelize=parallelize, 
                         model_to_be_loaded = model_to_be_loaded,
                         classifier_variant=classifier_variant)
        self.task_type = "TC"

    def create_train_data_loader(self, sentence_triples: List[Tuple[str, str, str]],
                                 labels: List[int], shuffle, relations):
        """
        Creates a DataLoader for training on sentence triples.

        :param sentence_triples: A list of sentence triples.
        :param labels: A list of labels for each sentence triple.
        :param shuffle: Whether to shuffle the data.
        :param relations: Relationship labels for each sentence triple.
        :return: A DataLoader for the training data.
        """

        dataset = SentenceTripleDataset(sentence_triples, self.tokenizer,
                                        max_length=BERT_MAX_SEQUENCE_LENGTH,
                                        labels=labels, relations=relations)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def create_test_data_loader(self, sentence_triples: List[Tuple[str, str, str]], shuffle):
        """
        Creates a DataLoader for testing on sentence triples.

        :param sentence_triples: A list of sentence triples for testing.
        :param shuffle: Whether to shuffle the test data.
        :return: A DataLoader for the test data.
        """

        dataset = SentenceTripleDataset(sentence_triples, self.tokenizer,
                                        max_length=BERT_MAX_SEQUENCE_LENGTH)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def predict(self, sentence_triples: List[Tuple[str, str, str]], name_test_set,
                model_name, batch_size):
        """
        Makes predictions on a set of sentence triples.

        :param sentence_triples: A list of sentence triples to predict on.
        :param name_test_set: The name of the test set.
        :param model_name: The name of the model.
        :param batch_size: The batch size for prediction.
        :return: A DataFrame with predictions and probabilities for each relation.
        """

        self.model.eval()

        self.batch_size = batch_size
        data_loader = self.create_test_data_loader(sentence_triples, shuffle=False)

        all_logits = []
        all_predictions = []

        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predictions = torch.max(logits.data, 1)
                all_logits.extend(logits.tolist())
                all_predictions.extend(predictions.tolist())

        df_predictions = pd.DataFrame(sentence_triples, columns=["head", "relation", "tail"])
        df_predictions['prediction'] = all_predictions
        probits = [softmax(logits) for logits in all_logits]
        df_predictions['probits'] = [[round(num, 3) for num in sublist] for sublist in probits]

        pivot_df = df_predictions.pivot(index=['head', 'tail'], columns='relation',
                                        values='probits')

        probs_df = pivot_df.apply(lambda x: x.map(lambda y: y[1]), axis=1)
        final_df = probs_df.reset_index()

        if "WN18RR" in name_test_set:
            relation_list = WN18RR_RELATIONS
        elif ("CNL" in name_test_set) or ("ESC0" in name_test_set) or ("handcrafted" in name_test_set):
            relation_list = ESCO_CNL_RELATIONS
        elif "stroma" in name_test_set:
            relation_list = STROMA_DATA_RELATIONS
        else:
            sys.exit("Could not select relation list. Check function 'predict' in BERT TC classifier if still up to date with the used datasets")

        final_df['highest_prob'] = final_df[relation_list].idxmax(axis=1)
        
        save_processed_df_to_csv(final_df, f"TC_{model_name}_{name_test_set}_prediction", "predictions")

        return final_df

    def evaluate(self, test_df, sentence_triples: List[Tuple[str, str, str]], labels: List[int],
                  name_test_set, model_name, batch_size):
        """
        Evaluates the model on a set of sentence triples and labels.

        :param test_df: A DataFrame containing the test data and labels.
        :param sentence_triples: A list of sentence triples.
        :param labels: The true labels for each sentence triple.
        :param name_test_set: The name of the test set.
        :param model_name: The name of the model.
        :param batch_size: The batch size for evaluation.
        :return: The accuracy of the predictions.
        """

        self.model.eval()
        
        final_df = self.predict(sentence_triples, name_test_set, model_name, batch_size)

        test_df_true = test_df[test_df['label'] == 1]
        merged_df = pd.merge(final_df, test_df_true[['head', 'tail', 'relation']],
                             on=['head', 'tail'], how='left')
        merged_df.rename(columns={'relation': 'actual_relation'}, inplace=True)

        accuracy = accuracy_score(merged_df['actual_relation'], merged_df['highest_prob'])
        print('Overall Accuracy: %d %%' % (100 * accuracy))

        if "WN18RR" in name_test_set:
            label_dict = WN18RR_LABEL_DICT
        elif ("CNL" in name_test_set) or ("ESC0" in name_test_set) or ("handcrafted" in name_test_set):
            label_dict = ESCO_CNL_LABEL_DICT
        elif "stroma" in name_test_set:
            label_dict = STROMA_DATA_LABEL_DICT
        else:
            sys.exit("Could not select label dictionary. Check function 'evaluate' in BERT TC classifier if still up to date with the used datasets")

        target_names = list(label_dict.values())
        cm: np.ndarray = confusion_matrix(merged_df['actual_relation'], merged_df['highest_prob'])
        cm_df: pd.DataFrame = pd.DataFrame(cm, index=[f'Actual {label}' for label in target_names],
                                        columns=[f'Predicted {label}' for label in target_names])
        print("Confusion Matrix:")
        print(cm_df)
        print("Classification Report:")
        target_names = [label_dict[i] for i in range(len(label_dict))]
        print(classification_report(merged_df['actual_relation'], merged_df['highest_prob'],
                                    target_names=target_names))
        
        res_to_save = ['Overall Accuracy: %d %%' % (100 * accuracy), "Confusion Matrix:", cm_df,
                       "Classification Report:", classification_report(merged_df['actual_relation'],
                                                                       merged_df['highest_prob'],
                                                                       target_names=target_names)]
        save_printed_output_to_file(res_to_save, f"TC_{model_name}_{name_test_set}",
                                    "evaluation_results")