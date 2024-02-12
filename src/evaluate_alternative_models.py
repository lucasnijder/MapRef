import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from utils import *

def load_predictions(args):
    """
    Loads the prediction data based on the specified model and test set.

    :param args: Command-line arguments containing model and test_set information.
    :type args: argparse.Namespace

    :return: A pandas DataFrame containing the prediction data with columns ['head', 'tail', 'relation'].
    :rtype: pandas.DataFrame
    """
    if args.model == "stroma":
        pred_df = read_txt_to_df(f"STROMA_{args.test_set}",
                                 source_folder="processed/predictions")
        pred_df = pred_df[0].str.split(' :: ', expand=True)
        pred_df.columns = ['head','tail','relation']

        pred_df['relation'] = pred_df['relation'].astype(int)

        pred_df['relation'] = pred_df['relation'].map(STROMA_MODEL_LABEL_DICT)
    elif "GPT" in args.model:
        pred_df = read_processed_csv(f"{args.model}_{args.test_set}", 
                                     "predictions")
        print(len(pred_df))
        if len(pred_df.columns) == 3:
            pred_df.columns = ['head','tail','relation']#,'response_text']
        if len(pred_df.columns) == 4:
            pred_df.columns = ['head','tail','relation','response_text']
        print(pred_df)

    else:
        sys.exit("Given model not known, please retry with different model name")

    pred_df = pred_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    return pred_df

def get_ground_truth(args):
    """
    Retrieves the ground truth data for the specified test set and anchor configuration.

    :param args: Command-line arguments containing test_set and anchors information.
    :type args: argparse.Namespace

    :return: A pandas DataFrame of the ground truth data.
    :rtype: pandas.DataFrame
    """
    if args.anchors == "default":
        ground_truth = read_processed_csv(f"RP_test_{args.test_set}",
                                          f"test/default/{args.test_set}")
    else:
        ground_truth = read_processed_csv(f"RP_test_{args.test_set}",
                                          f"test/anchors_{args.anchors}/{args.test_set}")

    return ground_truth

def evaluate(args, pred, ground_truth, print_bool=True):
    """
    Evaluates the predictions against the ground truth data, calculating accuracy, confusion matrix, and classification report.

    :param args: Command-line arguments for additional context (unused in the function but may be used for extensions).
    :type args: argparse.Namespace
    :param pred: DataFrame containing the predictions.
    :type pred: pandas.DataFrame
    :param ground_truth: DataFrame containing the ground truth data.
    :type ground_truth: pandas.DataFrame
    :param print_bool: Boolean flag to control the printing of evaluation metrics, defaults to True.
    :type print_bool: bool

    :return: A tuple containing overall accuracy and macro-average F1 score.
    :rtype: tuple
    """
    relations = sorted(pd.concat([pred['relation'], 
                                  ground_truth['relation']]).drop_duplicates().reset_index(drop=True))

    print(len(ground_truth))
    print(len(pred))

    accuracy = accuracy_score(ground_truth['relation'], pred['relation'])
    if print_bool:
        print('Overall Accuracy: %d %%' % (100 * accuracy))

    cm: np.ndarray = confusion_matrix(ground_truth['relation'], 
                                      pred['relation'])
    cm_df: pd.DataFrame = pd.DataFrame(cm, 
                                       index=relations, 
                                       columns=relations)
    if print_bool:
        print("Confusion Matrix:")
        print(cm_df)

    class_report = classification_report(ground_truth['relation'], 
                                         pred['relation'], 
                                         target_names=relations, 
                                         output_dict=True)
    macro_avg_f1 = class_report['macro avg']['f1-score']
    if print_bool:
        print("Classification Report:")
        print(classification_report(ground_truth['relation'], pred['relation'],
                                     target_names=relations))

    return accuracy, macro_avg_f1

def main(args) -> None:
    """
    Main function to orchestrate the model evaluation process. It loads predictions, retrieves ground truth data, and evaluates the model.

    :param args: Parsed command-line arguments specifying the test set, anchor configuration, and model used for predictions.
    :type args: argparse.Namespace
    """
    pred = load_predictions(args)

    print(pred['relation'].unique())

    ground_truth = get_ground_truth(args)

    print(ground_truth['relation'].unique())

    evaluate(args, pred, ground_truth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--test_set", type=str, required=False,
                        help="Choose which test set to evaluate")
    parser.add_argument("--anchors", type=str, required=True,
                        help="Choose how many achors were used")
    parser.add_argument("--model", type=str, required=False,
                        help="Choose which model was used")
    args = parser.parse_args()

    main(args)
