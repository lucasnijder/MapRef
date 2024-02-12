import time
import argparse
import sys
import re

import wandb

# Import all functions from the utils.py file
from utils import *


def run_ESCO_CNL_case():
    """
    This function executes the data preparation steps for ESCO and CNL datasets.
        It runs the scripts for processing the raw data and converting them
        into the correct format for training and evaluating.
    """

    print("**********************************")
    print("    create train sets ESCO,CNL        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "CNL_ontology_to_triples"])
    run_script("data_preparation.py", ["--action", "ESCO_ontology_to_triples"])
    run_script("data_preparation.py", ["--action", "train_set_converter",
                                       "--data_sets", "CNL,ESCO",
                                       "--task_type","RP",
                                       "--balanced","F"])

    print("**********************************")
    print("    create test sets ESCO,CNL        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "ESCO-CNL_mapping_to_triples"])
    run_script("data_preparation.py", ["--action", "test_set_converter",
                                       "--data_sets", "ESCO-CNL",
                                       "--task_type", "RP"]) 

def run_ESCO_ONET_case():
    """
    This function executes the data preparation steps for ESCO and O*NET datasets.
        It runs the scripts for processing the raw data and converting them
        into the correct format for training and evaluating.
    """

    print("**********************************")
    print("    create train sets ESCO,ONET        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "ONET_ontology_to_triples"])
    run_script("data_preparation.py", ["--action", "ESCO_ontology_to_triples"])
    run_script("data_preparation.py", ["--action", "train_set_converter",
                                       "--data_sets", "ONET,ESCO",
                                       "--task_type","RP",
                                       "--balanced","F"])

    print("**********************************")
    print("    create test sets ESCO,ONET        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "ESCO-ONET_mapping_to_triples"])
    run_script("data_preparation.py", ["--action", "test_set_converter",
                                       "--data_sets", "ESCO-ONET",
                                       "--task_type", "RP"]) 


def run_stroma_case():
    """
    This function executes the data preparation steps for the stroma datasets.
        It runs the scripts for processing the raw data and converting them
        into the correct format for training and evaluating.
    """

    print("**********************************")
    print("   create train sets stroma        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "stroma_ontologies_to_triples"])
    
    for d_name in ["g2","g4","g5","g6"]:
        run_script("data_preparation.py", ["--action", "train_set_converter",
                                           "--data_sets", f"stroma_{d_name}_source,stroma_{d_name}_target",
                                           "--task_type", "RP",
                                           "--balanced", "F"])

    print("**********************************")
    print("    create test sets stroma       ")
    print("**********************************")
    for d_name in ["g2","g4","g5","g6"]:
        run_script("data_preparation.py", ["--action", "stroma_mappings_to_triples"])
        run_script("data_preparation.py", ["--action", "test_set_converter",
                                           "--data_sets", f"stroma_{d_name}_reference",
                                           "--task_type", "RP"]) 

def run_bio_cases():
    """
    This function executes the data preparation steps for the biomedical datasets.
        It runs the scripts for processing the raw data and converting them
        into the correct format for training and evaluating.
    """

    print("**********************************")
    print("    create train sets BIO        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "bio_ontologies_to_triples"])
    
    for case in ['ncit,doid',
                 'omim,ordo',
                 'snomed.body,fma.body',
                 'snomed.neoplas,ncit.neoplas',
                 'snomed.pharm,ncit.pharm']:
        run_script("data_preparation.py", ["--action","train_set_converter",
                                             "--data_sets", case,
                                             "--task_type", "RP",
                                             "--balanced", "F"])

    print("**********************************")
    print("    create test sets BIO        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "bio_mappings_to_triples"])

    for case in ['ncit-doid',
                 'omim-ordo',
                 'snomed.body-fma.body',
                 'snomed.neoplas-ncit.neoplas',
                 'snomed.pharm-ncit.pharm']:
        run_script("data_preparation.py", ["--action", "test_set_converter",
                                           "--data_sets", case,
                                           "--task_type", "RP"]) 

def run_all_preprocessing():
    """
    The function runs all data pre-processing functions for all use cases
    """

    # run_ESCO_CNL_case()
    run_ESCO_ONET_case()
    run_stroma_case()
    run_bio_cases()

def run_training():
    """
    This function can be used to train a mapping refinement model. Make sure
        the data is processed correctly and converted to the correct format
    """

    run_script("train.py", ["--model_variant", "dwsunimannheim/TaSeR",   # distilbert-base-uncased
                            "--classifier_variant", "distilbert",   # distilbert
                            "--task_type", "RP",
                            "--train_set_names", "stroma_g6_source,stroma_g6_target", 
                            "--train_set_balanced", "F",
                            "--anchors", "default",
                            "--num_labels", "3",
                            "--lr", "0.00001",
                            "--epochs", "3",
                            "--batch_size", "8",
                            "--parallelize", "F",
                            "--balance_class_weights", "T"])
    
def run_model_test():
    """
    This function can be used to test an earlier trained model
    """

    run_script("evaluate.py", ["--model_variant", "dwsunimannheim/TaSeR",   # distilbert-base-uncased
                            "--classifier_variant", "taser",   # distilbert
                            "--eval_type", "test",
                            "--dataset_name", "stroma_g6_source,stroma_g6_target",
                            "--anchors", "default",
                            "--task_type", "RP",
                            "--model_name", "philip475",
                            "--num_labels", "3",
                            "--taser_default_model", "F"])

def run_train_and_eval(model_variant, classifier_variant, task_type,
                       train_set_names, train_set_balanced, anchors, eval_type,
                        test_set, num_labels, lr, epochs, batch_size,
                        parallelize, balance_class_weights):
    """
    Trains and evaluates a model based on specified parameters and datasets, 
    then logs the evaluation results to Wandb.

    :param model_variant: The variant of the model to be trained.
    :param classifier_variant: The classifier variant to be used.
    :param task_type: The type of task (RP/TC).
    :param train_set_names: Names of the datasets used for training.
    :param train_set_balanced: Whether to use a balanced training set.
    :param anchors: Specification of anchors to be used.
    :param eval_type: Whether to use validation set or test set for evaluation.
    :param test_set: The dataset used for evaluation.
    :param num_labels: The number of classes of the classification task.
    :param lr: Learning rate.
    :param epochs: The number epochs.
    :param batch_size: The size of each batch.
    :param parallelize: Whether to parallelize training across multiple GPUs.
    :param balance_class_weights: Whether to balance class weights in loss
                                    computation.

    :returns: A tuple containing the model name, validation accuracy, 
                weighted average F1 score, and macro average F1 score.
    :rtype: tuple
    """

    start_time = time.time()

    run_script("train.py", ["--model_variant", model_variant,
                            "--classifier_variant", classifier_variant,
                            "--task_type", task_type,
                            "--train_set_names", train_set_names, 
                            "--train_set_balanced", train_set_balanced,
                            "--anchors", "default",
                            "--num_labels", num_labels,
                            "--lr", lr,
                            "--epochs", epochs,
                            "--batch_size", batch_size,
                            "--parallelize", parallelize,
                            "--balance_class_weights", balance_class_weights,
                            ])
    
    with open("../prints/miscellaneous/run_info.txt", "r",
              encoding='cp1252') as file:
        run_info = file.readlines()[-1].strip()
        model_name = run_info.split(';')[0]
        run_id = run_info.split(';')[1]

    run_script("evaluate.py", ["--model_variant", model_variant, 
                            "--eval_type", eval_type,
                            "--dataset_name", test_set,
                            "--anchors", anchors,
                            "--task_type", task_type,
                            "--model_name", model_name,
                            "--num_labels", num_labels,
                            "--classifier_variant", classifier_variant])

    save_path = f"../prints/evaluation_results/{task_type}_{model_name}_{test_set}_{eval_type}.txt"
    with open(save_path, 'r', encoding='cp1252') as file:
        file_content = file.read()
    accuracy_match = re.search(r"Overall Accuracy: (\d+ %)", file_content)
    if accuracy_match:
        validation_accuracy = float(accuracy_match.group(1)[:-2])/100
        print("Accuracy:", validation_accuracy)

    f1_weighted_regex = r"weighted avg\s+\S+\s+\S+\s+(\S+)"
    f1_score_weighted_avg_match = re.search(f1_weighted_regex,
                                             file_content)
    if f1_score_weighted_avg_match:
        validation_weighted_avg_f1_score = f1_score_weighted_avg_match.group(1)
        print("Weighted Average F1 Score:", validation_weighted_avg_f1_score)

    f1_macro_regex = r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+"
    f1_score_macro_avg_match = re.search(f1_macro_regex,
                                         file_content)
    if f1_score_macro_avg_match:
        validation_macro_avg_f1_score = f1_score_macro_avg_match.group(1)
        print("Macro Average F1-score Score:", validation_macro_avg_f1_score)

    duration = time.time() - start_time

    wandb.init(project="clean_up", id=run_id, resume="must")
    wandb.log({"validation_accuracy": validation_accuracy})
    wandb.log({"validation_weighted_avg_f1_score": validation_weighted_avg_f1_score})
    wandb.log({"validation_macro_avg_f1_score": validation_macro_avg_f1_score})
    wandb.log({"train_test_time": duration})
    wandb.finish()

    return (model_name, float(validation_accuracy), 
            float(validation_weighted_avg_f1_score),
            float(validation_macro_avg_f1_score))

def run_exp1():
    """
    Executes Experiment 1, running training and evaluation cycles for 
        different model and classifier variants across specified datasets.

    This function iterates through pre-defined training set options, 
        adjusting batch sizes based on the dataset characteristics.
        It then trains and evaluates models using the run_train_and_eval
        function for each combination of model name and classifier variant
        specified within the loop. 
    """

    train_set_names_options = ['stroma_g4_source,stroma_g4_target']   # ALL_USE_CASES
    test_set_options = MAP_ALL_TRAIN_TO_TEST
    for train_set in train_set_names_options:
        test_set = test_set_options[train_set]

        if "stroma" in train_set:
            batch_size = 8
        elif "ESCO" in train_set:
            batch_size = 64
        else:
            batch_size = 128

        for model_name, classifier_variant in [['dwsunimannheim/TaSeR', "taser"],
                                               ['distilbert-base-uncased', 'distilbert']]:
            model_name, _, _, _ = run_train_and_eval(
                model_variant = model_name,
                classifier_variant = classifier_variant,
                task_type = "RP",
                train_set_names = train_set,
                train_set_balanced = "F",
                anchors = "default",
                eval_type= "test",
                test_set = test_set,
                num_labels = str(3),
                lr = str(0.00001),
                epochs = str(0),
                batch_size = str(batch_size),
                parallelize = "F",
                balance_class_weights = "T")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True)
    args = parser.parse_args()

    if args.exp == "data_all":
        run_all_preprocessing()
    elif args.exp == 'data_ESCO-CNL':
        run_ESCO_CNL_case()
    elif args.exp == "data_ESCO-ONET":
        run_ESCO_ONET_case()
    elif args.exp == 'data_stroma':
        run_stroma_case()
    elif args.exp == "data_bio":
        run_bio_cases()
    elif args.exp == "train":
        run_training()
    elif args.exp == "test":
        run_model_test()
    elif args.exp == "1":
        run_exp1()
    else:
        sys.exit(f"Unknown experiment: {args.exp}. Try again")
