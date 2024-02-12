import argparse

import numpy
from torch.backends import cudnn
from sklearn.utils.class_weight import compute_class_weight
import wandb

from classifier import BertClassifier_relation_prediction, BertClassifier_triple_classification
from utils import *

def initialize_classifier(args) -> object:
    """
    Initializes and returns a classifier object based on the task type and provided arguments.

    :param args: Command-line arguments containing model configuration and options.
    :type args: argparse.Namespace

    :returns: An instance of a classifier, either for relation prediction or triple classification.
    :rtype: object
    """

    common_args = {
        'model_variant': args.model_variant,
        'num_labels': args.num_labels,
        'parallelize': args.parallelize,
        'model_to_be_loaded': args.general_model,
        'classifier_variant': args.classifier_variant
    }
    classifier = None
    if args.task_type == "RP":
        classifier = BertClassifier_relation_prediction(**common_args)
    elif args.task_type == "TC":
        classifier = BertClassifier_triple_classification(**common_args)
    return classifier

def set_wandb_config(classifier, args, sorted_train_set_names, custom_name):
    """
    Sets the configuration for the Weights & Biases (wandb) logging based on the classifier and arguments.

    :param classifier: The classifier object for which the wandb configuration is to be set.
    :type classifier: object
    :param args: Command-line arguments containing model and training configuration.
    :type args: argparse.Namespace
    :param sorted_train_set_names: A string of sorted training set names.
    :type sorted_train_set_names: str
    :param custom_name: Custom name for the wandb run.
    :type custom_name: str
    """

    safe_model_variant = args.model_variant.replace("/", "-")
    classifier.wandb = wandb.init(project="clean_up",
                                  name=custom_name,
                                  entity="lucas-snijder-tno")
    wandb.config.update({
        'task': args.task_type,
        'model': safe_model_variant,
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'parallelized': args.parallelize,
        'train_set': sorted_train_set_names,
        'anchors' : args.anchors,
        'max_sequence_length':BERT_MAX_SEQUENCE_LENGTH,
    })

def read_data(args, sorted_train_set_names):
    """
    Reads and returns training data, labels, and relations (if applicable) based on the task type and training set names.

    :param args: Command-line arguments containing model and data configuration.
    :type args: argparse.Namespace
    :param sorted_train_set_names: A string of sorted training set names for identifying the data files.
    :type sorted_train_set_names: str

    :returns: A tuple containing training sentences, labels, and relations (None if not applicable).
    :rtype: tuple
    """

    data_type = args.task_type
    if data_type == "TC" :
        item_type = "triples"
    elif data_type == "RP":
        item_type = "pairs"
    else:
        sys.exit("problem with reading data in train.py script")

    train_sentences = read_processed_var_from_pickle(
        f"{data_type}_train_{item_type}_{sorted_train_set_names}",
        f"train/{args.anchors}/{sorted_train_set_names[:-3]}")
    train_labels = read_processed_var_from_pickle(
        f"{data_type}_train_labels_{sorted_train_set_names}", 
        f"train/{args.anchors}/{sorted_train_set_names[:-3]}")

    if data_type == "TC":
        train_relations = read_processed_var_from_pickle(
            f"{data_type}_train_relations_{sorted_train_set_names}",
            f"train/{args.anchors}/{sorted_train_set_names[:-3]}")
    else:
        train_relations = None

    return train_sentences, train_labels, train_relations

def get_relations_class_weight_matrix(args, train_sentences, train_labels):
    """
    Calculates and returns a class weight matrix for relation classes based on the training data.
        Can be used by specifying the --balance_class_weights argument as "matrix"

    :param args: Command-line arguments containing model and data configuration.
    :type args: argparse.Namespace
    :param train_sentences: Training sentences.
    :type train_sentences: List
    :param train_labels: Training labels.
    :type train_labels: List

    :returns: A matrix of class weights for true and false relations.
    :rtype: list
    """

    t_rel_dict = {}
    for t, l in zip(train_sentences, train_labels):
        if l == 1:
            if t[1] in t_rel_dict:
                t_rel_dict[t[1]] += 1
            else:
                t_rel_dict[t[1]] = 1

    f_rel_dict = {}
    for t, l in zip(train_sentences, train_labels):
        if l == 0:
            if t[1] in f_rel_dict:
                f_rel_dict[t[1]] += 1
            else:
                f_rel_dict[t[1]] = 1

    total_t = np.sum(list(t_rel_dict.values()))
    total_f = np.sum(list(f_rel_dict.values()))

    for key in t_rel_dict:
        t_rel_dict[key] = total_t/t_rel_dict[key]

    for key in f_rel_dict:
        f_rel_dict[key] = total_f/f_rel_dict[key]

    replacement_dict = {'broadmatch':0, 'exactmatch':1, 'narrowmatch':2}

    t_rel_dict = {replacement_dict.get(key, key): value for key, value in t_rel_dict.items()}
    f_rel_dict = {replacement_dict.get(key, key): value for key, value in f_rel_dict.items()}

    t_rel_dict = [value for key, value in sorted(t_rel_dict.items())]
    f_rel_dict = [value for key, value in sorted(f_rel_dict.items())]
               
    class_weights_matrix = [f_rel_dict, t_rel_dict]

    # divide by mean of list for easier weights
    for i, weights in enumerate(class_weights_matrix):
        class_weights_matrix[i] = scale_list_by_mean(weights)

    return(class_weights_matrix)

def handle_class_weights(args, train_sentences, train_labels):
    """
    Determines and returns the class weights based on 
        the training labels and specified balancing strategy.

    :param args: Command-line arguments specifying whether and how to balance class weights.
    :type args: argparse.Namespace
    :param train_sentences: Training sentences, used for calculating relation class weights.
    :type train_sentences: List
    :param train_labels: Training labels, used for calculating class weights.
    :type train_labels: List

    :returns: Class weights, either as a list or a matrix, or None if no balancing is applied.
    :rtype: list or None
    """

    if args.balance_class_weights == 'T':
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(train_labels), 
                                             y=train_labels)
    elif args.balance_class_weights == "matrix":
        class_weights = get_relations_class_weight_matrix(args, 
                                                          train_sentences, 
                                                          train_labels)
    else:
        class_weights = None
    return class_weights

def main(args) -> None:
    """
    The main function to execute the training workflows based on command-line arguments.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    """

    cudnn.benchmark = True

    classifier = initialize_classifier(args)
    sorted_train_set_names = sort_words_str(args.train_set_names)

    if args.train_set_balanced == 'T':
        sorted_train_set_names += '_BA'
    elif args.train_set_balanced == 'F':
        sorted_train_set_names += '_UB'
    else:
        sys.exit('Unknown value for train_set_balanced. Choose from T or F.')

    custom_name = random_name_generator()
    set_wandb_config(classifier, args, sorted_train_set_names, custom_name)

    train_sentences, train_labels, train_relations = read_data(args, sorted_train_set_names)

    class_weights = handle_class_weights(args, train_sentences, train_labels)

    classifier.train(train_sentences,
                     train_labels,
                     lr=args.lr,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     class_weights=class_weights,
                     relations=train_relations)

    print(args.model_variant.replace("/", "-"), custom_name, classifier)
    save_model_to_pth(args.model_variant.replace("/", "-"), custom_name, classifier)

    with open("../prints/miscellaneous/run_info.txt", "a",encoding='cp1252') as file:
        file.write(custom_name + ";" + classifier.wandb.id + "\n")

    classifier.wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_variant", required=True, type=str, 
                        help="Choose which HF model variant to use.")
    parser.add_argument("--classifier_variant", required=True, type=str, 
                        help="Choose whether to use BERTForSequenceClassification (default) or Demir's classifier (demir).")
    parser.add_argument("--task_type", required=True, type=str, 
                        help="Choose which task type to train the model for. Choices are RP and TC ")
    parser.add_argument("--train_set_names", required=True, type=str, 
                        help="Indicate which data sets to use for training. Specify using a comma without spaces between them.")
    parser.add_argument("--train_set_balanced", required=True, type=str, 
                        help="Indicate wheter you want to use a train set of which the classes have been balanced. For TC this concerns the relations, not the labels.")
    parser.add_argument("--anchors", required=True, type=str, 
                        help="Indicate the percentage of anchors to use, for no anchors set to default")    
    parser.add_argument("--num_labels", required=True, type=int, 
                        help="Indicate how many labels the data for the problem at hand has. For TC, the number is 2.")
    parser.add_argument("--balance_class_weights", required=False, type=str, 
                        help="Indicate whether you want the model to take into account class imbalances.")
    parser.add_argument("--lr", required=True, type=float, 
                        help="Specify the learning rate.")
    parser.add_argument("--epochs", required=True, type=int,
                        help="Specify the number of epochs.")
    parser.add_argument("--batch_size", required=True, type=int, 
                        help="Specify the batch size.")
    parser.add_argument("--parallelize", required=True, type=str, 
                        help="Indicate whether you want to use parallel computing across multiple GPUs. The visible GPUs are specified in the train.py file.")
    parser.add_argument("--general_model", required=False, type=str, default=None, 
                        help="If you want to further fine-tune an already fine-tuned model, specify the model-name here.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
