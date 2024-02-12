import argparse
from typing import List, Tuple, Dict
from classifier import BertClassifier_relation_prediction, BertClassifier_triple_classification
import warnings
from utils import *

def get_label_dict(args) -> Dict[int, str]:
    """
    Determines the label dictionary to use based on the dataset name and other conditions specified in the arguments.

    :param args: Command-line arguments including dataset name, taser default model flag, number of labels, and task type.
    :type args: argparse.Namespace
    
    :return: The appropriate label dictionary for mapping integer labels to their string representations.
    :rtype: Dict[int, str]
    """
    if ("CNL" in args.dataset_name) or (
            "ESCO" in args.dataset_name) or (
                "handcrafted" in args.dataset_name) or (
                    "ncit" in args.dataset_name) or (
                        "omim" in args.dataset_name) or (
                            "snomed" in args.dataset_name):
        label_dict = ESCO_CNL_LABEL_DICT
    elif "stroma" in args.dataset_name:
        label_dict = STROMA_DATA_LABEL_DICT
    else:
        sys.exit(
            "Could not select relation list, check function "
            "'get_label_dict' if still up to date with the used datasets"
        )

    if (args.taser_default_model == "T") or (args.num_labels == 7):
        label_dict = TASER_LABEL_DICT

    if args.num_labels == 2 and args.task_type == "TC":
        return FALSE_TRUE_LABEL_DICT
    
    print(label_dict)

    return label_dict

def get_classifier_and_test_data(args) -> Tuple:
    """
    Initializes the appropriate classifier and loads test data based on task type and evaluation settings provided in the arguments.

    :param args: Command-line arguments specifying evaluation type, dataset name, anchors, model variant, classifier variant, model name, and number of labels.
    :type args: argparse.Namespace

    :return: A tuple containing the initialized classifier, test pairs, test labels, and test data (if task type is TC).
    :rtype: Tuple
    """
    classifiers = {
        "RP": BertClassifier_relation_prediction,
        "TC": BertClassifier_triple_classification,
    }
    dict_task_type_data = {
        "TC":"TC",
        "RP":"RP",
    }
    dict_data_type = {
        "TC":"triples",
        "RP":"pairs",
    }
    dict_dataset_names = {
        "ESCO-CNL":"CNL_ESCO",
        "stroma_g2_reference":"stroma_g2_source_stroma_g2_target",
        "stroma_g4_reference":"stroma_g4_source_stroma_g4_target",
        "stroma_g5_reference":"stroma_g5_source_stroma_g5_target",
        "stroma_g6_reference":"stroma_g6_source_stroma_g6_target",
        "stroma_g7_reference":"stroma_g7_source_stroma_g7_target",
    }

    if args.eval_type == "validation":
        args.dataset_name = dict_dataset_names[args.dataset_name]

    task_type_data = dict_task_type_data[args.task_type]
    data_type = dict_data_type[args.task_type]

    classifier = classifiers[args.task_type](
        model_variant=args.model_variant,
        num_labels=args.num_labels,
        model_to_be_loaded=args.model_name,
        classifier_variant=args.classifier_variant)
    test_pairs = read_processed_var_from_pickle(
        f"{task_type_data}_{args.eval_type}_{data_type}_{args.dataset_name}", 
        f"{args.eval_type}/{args.anchors}/{args.dataset_name}")
    test_labels = read_processed_var_from_pickle(
        f"{task_type_data}_{args.eval_type}_labels_{args.dataset_name}",
        f"{args.eval_type}/{args.anchors}/{args.dataset_name}")
    test_data = None
    
    if args.task_type == "TC":
        test_data = read_processed_csv(
            f"TC_{args.eval_type}_{args.dataset_name}",
            f"{args.eval_type}/{args.anchors}/{args.dataset_name}")
        
    return classifier, test_pairs, test_labels, test_data

def main(args) -> None:
    """
    Main function to perform model evaluation. It retrieves the correct label dictionary, initializes the classifier, loads test data, and performs evaluation.

    :param args: Parsed command-line arguments specifying details about the model variant, classifier variant, model name, evaluation type, dataset name, anchors, number of labels, task type, and whether the taser default model is used.
    :type args: argparse.Namespace
    """
    label_dict = get_label_dict(args)
    if label_dict is None:
        sys.exit("Unknown dataset, check main function in evaluate.py")
        
    classifier, test_pairs, test_labels, test_data = get_classifier_and_test_data(args)
    
    full_dataset_name = args.dataset_name + "_" + args.eval_type

    if args.task_type == "TC":
        classifier.evaluate(test_data, test_pairs, test_labels, 
                            full_dataset_name, args.model_name, batch_size=128)
    else:
        if args.num_labels == 2:
            warnings.warn("The task type is RP and number"
                          "of labels is set to 2, is this correct?")
        classifier.evaluate(test_pairs, test_labels, 
                            label_dict, full_dataset_name, 
                            args.model_name, batch_size=128)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model_variant", type=str, required=True,
                        help="Choose which HF model variant to use.")
    parser.add_argument("--classifier_variant", type=str, required=True,
                        help="Choose whether to use BERTForSequenceClassification (default) or Demir's classifier (demir).")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Specify which model should be evaluated")
    parser.add_argument("--eval_type", type=str, required=True, 
                        help="Specify whether you want to use a validation or a test set")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Give the name of the data set that should be used for evaluation.")
    parser.add_argument("--anchors", required=True, type=str, 
                        help="Indicate the percentage of anchors used, for no anchors set to default")    
    parser.add_argument("--num_labels", type=int, required=True, 
                        help="Specify the number of labels. If task_type TC or NewTC is used this should be set to 2.")
    parser.add_argument("--task_type", type=str, required=True, 
                        help="Specify what task type the to-be-evaluatd model is trained for. Choices are RP, NewRP, TC, NewTC")
    parser.add_argument("--taser_default_model", type=str, required=False, 
                        help="Specify whether the stock taser variant is used. Optional")
    args = parser.parse_args()

    main(args)
