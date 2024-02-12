import pandas as pd
import os
import sys
import argparse
from openai import OpenAI
import pandas as pd
from transformers import AutoTokenizer
import transformers
import torch
import re
import time

from utils import *

os.environ["OPENAI_API_KEY"] = "xxx"
client = OpenAI()


def get_prompt_format(type, concept1 = "placeholder1", concept2 = "placeholder2", examples=None, model="GPT4"):
    """
    Generates a prompt format for the relation prediction task based on the specified settings.

    :param type: The type of prompt to generate, such as 'system', 'user_one_shot', or 'user_few_shot'.
    :type type: str
    :param concept1: The first concept for relation prediction, defaults to 'placeholder1'.
    :type concept1: str, optional
    :param concept2: The second concept for relation prediction, defaults to 'placeholder2'.
    :type concept2: str, optional
    :param examples: A list of examples for few-shot learning, defaults to None.
    :type examples: List[Tuple[str, str, str]], optional
    :param model: Specifies the model version to use, defaults to 'GPT4'.
    :type model: str, optional

    :return: A formatted prompt string based on the input parameters.
    :rtype: str
    """
    if model == "GPT35" or model == "GPT4":
        if type == "system":
            return "You are a relation prediction expert. You know the relation between two given concepts. The choices of relations are exactmatch, narrowmatch, and broadmatch. Only output the predicted relation without any extra words, characters, or symbols."
        elif type == "user_one_shot":
            return f"Determine the relation between the concepts: [{concept1}],  and: [{concept2}]"
        elif type == "user_few_shot":
            example_str = ""
            for exp in examples:
                example_str = example_str + f'Relation between concepts: [{exp[0]}], and: [{exp[2]}] is {exp[1]}\n'

            return f'{example_str}Determine the relation between the concepts: [{concept1}],  and: [{concept2}] '


class LLMRefiner():
    """
    Defines a base class for refining predictions using Large Language Models (LLMs) such as GPT-3.5 or GPT-4.
    """
    def get_prompt(self, concept1, concept2):
        """
        Generates system and user prompts for relation prediction based on the specified concepts and current setting.

        :param concept1: The first concept in the relation.
        :type concept1: str
        :param concept2: The second concept in the relation.
        :type concept2: str

        :return: System and user prompts for querying the LLM.
        :rtype: Tuple[str, str]
        """
        if self.setting == "one-shot":
            return get_prompt_format("system"), get_prompt_format("user_one_shot", concept1, concept2)
        elif self.setting == "few-shot":
            train_set = read_processed_csv(f'{self.anchors}/{self.usecase_name}/RP_train_{self.usecase_name}_UB','train')

            def get_match(concept, relation, is_head=True):
                condition = (train_set['head'] == concept) if is_head else (train_set['tail'] == concept)
                filtered_set = train_set.loc[condition & (train_set['relation'] == relation)]
                return tuple(filtered_set.iloc[0, :]) if not filtered_set.empty else None

            broadmatch_source = get_match(concept1, "broadmatch")
            narrowmatch_source = get_match(concept1, "narrowmatch", is_head=False)
            exactmatch_source = get_match(concept1, "exactmatch")

            broadmatch_target = get_match(concept2, "broadmatch")
            narrowmatch_target = get_match(concept2, "narrowmatch", is_head=False)
            exactmatch_target = get_match(concept2, "exactmatch")

            # Uncomment if needed and adjust accordingly
            parent_source = broadmatch_source[0] if broadmatch_source else None
            broadmatch_parent_source = get_match(parent_source, "broadmatch") if parent_source else None
            narrowmatch_parent_source = get_match(parent_source, "narrowmatch", is_head=False) if parent_source else None
            exactmatch_parent_source = get_match(parent_source, "exactmatch") if parent_source else None

            # Uncomment if needed and adjust accordingly
            parent_target = broadmatch_target[0] if broadmatch_target else None
            broadmatch_parent_target = get_match(parent_target, "broadmatch") if parent_target else None
            narrowmatch_parent_target = get_match(parent_target, "narrowmatch", is_head=False) if parent_target else None
            exactmatch_parent_target = get_match(parent_target, "exactmatch") if parent_target else None

            if broadmatch_source == None:
                filtered_set = train_set[train_set['relation']=='broadmatch'].sample(1)
                broadmatch_source = tuple(filtered_set.iloc[0, :])
            if broadmatch_target == None:
                filtered_set = train_set[train_set['relation']=='broadmatch'].sample(1)
                broadmatch_target = tuple(filtered_set.iloc[0, :])
            if broadmatch_parent_source == None:
                filtered_set = train_set[train_set['relation']=='broadmatch'].sample(1)
                broadmatch_parent_source = tuple(filtered_set.iloc[0, :])
            if broadmatch_parent_target == None:
                filtered_set = train_set[train_set['relation']=='broadmatch'].sample(1)
                broadmatch_parent_target = tuple(filtered_set.iloc[0, :])

            if narrowmatch_source == None:
                filtered_set = train_set[train_set['relation']=='narrowmatch'].sample(1)
                narrowmatch_source = tuple(filtered_set.iloc[0, :])
            if narrowmatch_target == None:
                filtered_set = train_set[train_set['relation']=='narrowmatch'].sample(1)
                narrowmatch_target = tuple(filtered_set.iloc[0, :])
            if narrowmatch_parent_source == None:
                filtered_set = train_set[train_set['relation']=='narrowmatch'].sample(1)
                narrowmatch_parent_source = tuple(filtered_set.iloc[0, :])
            if narrowmatch_parent_target == None:
                filtered_set = train_set[train_set['relation']=='narrowmatch'].sample(1)
                narrowmatch_parent_target = tuple(filtered_set.iloc[0, :])

            if exactmatch_source == None:
                filtered_set = train_set[train_set['relation']=='exactmatch'].sample(1)
                exactmatch_source = tuple(filtered_set.iloc[0, :])
            if exactmatch_target == None:
                filtered_set = train_set[train_set['relation']=='exactmatch'].sample(1)
                exactmatch_target = tuple(filtered_set.iloc[0, :])
            if exactmatch_parent_source == None:
                filtered_set = train_set[train_set['relation']=='exactmatch'].sample(1)
                exactmatch_parent_source = tuple(filtered_set.iloc[0, :])
            if exactmatch_parent_target == None:
                filtered_set = train_set[train_set['relation']=='exactmatch'].sample(1)
                exactmatch_parent_target = tuple(filtered_set.iloc[0, :])

            examples = [broadmatch_source, narrowmatch_source, exactmatch_source,
                        broadmatch_target, narrowmatch_target, exactmatch_target,]
                        #broadmatch_parent_source, narrowmatch_parent_source, exactmatch_parent_source,
                        #broadmatch_parent_target, narrowmatch_parent_target, exactmatch_parent_target]
            print(examples)

            random.shuffle(examples)

            
            system_prompt = get_prompt_format("system")
            user_prompt = get_prompt_format("user_few_shot", concept1, concept2, examples)
            return system_prompt, user_prompt

        else:
            sys.exit('unknown setting, choose from: one-shot, few-shot')

    def get_and_save_predictions(self):
        """
        Obtains predictions for each pair in the dataset and saves the responses.
        """
        preds = []
        response_text = []

        print(f"Total rows to process: {len(self.data)}")

        for idx, text in self.data[['head', 'tail','pred']].iterrows():
            if type(text['pred']) == float or text['pred'] == 'nan':
                print('yes')
                concept1 = text['head']
                concept2 = text['tail']
                pred, response_text = self.get_skos_relation(concept1, concept2)
                print(f"Processing row {idx}: {concept1}, {concept2} -> Prediction: {pred}")
                preds.append(pred)
                preds.append(response_text)
                self.data.iloc[idx, 2] = pred
                self.data.iloc[idx, 3] = response_text
                self.data.to_csv(self.pred_location, sep=';', index=False)

                print(f"Processed [{idx + 1}/{len(self.data)}] rows")

        print("Processing complete.")

    def get_skos_relation(self):
        """
        Placeholder method for getting SKOS relation predictions, to be implemented in subclasses.
        """
        return NotImplementedError
    
    def load_data(self):
        """
        Loads test data and prepares it for prediction refinement.
        """
        if os.path.exists(self.pred_location):
            self.data = pd.read_csv(self.pred_location, sep=';')
            print('yeah')
        else:
            self.data = read_processed_csv(f'RP_test_{self.testset_name}',f'test/{self.anchors}/{self.testset_name}')[['head','tail']]
            self.data['pred'] = ['nan']*len(self.data)
            self.data['response_text'] = ['nan']*len(self.data)

class GPTRefiner(LLMRefiner):
    """
    A subclass of LLMRefiner that specifically utilizes GPT models (e.g., GPT-3.5 or GPT-4) for refining predictions.
    """
    def __init__(self, testset_name, model, setting, anchors):
        """
        Initializes a GPTRefiner instance with specified parameters.

        :param testset_name: Name of the test set.
        :type testset_name: str
        :param model: Specifies which GPT model to use.
        :type model: str
        :param setting: Indicates the refinement setting ('one-shot' or 'few-shot').
        :type setting: str
        :param anchors: Specifies the anchor configuration used.
        :type anchors: str
        """
        self.testset_name = testset_name
        self.model = model
        self.data = None
        self.anchors = anchors
        self.setting = setting
        self.pred_location = f'../data/processed/predictions/{self.model}_{self.setting}_{str(TEMP)}_brackets_{self.testset_name}.csv'

        MAP_ALL_TRAIN_TO_TEST_inverse = {value: key for key, value in MAP_ALL_TRAIN_TO_TEST.items()}
        self.usecase_name = MAP_ALL_TRAIN_TO_TEST_inverse[testset_name].replace(',','_')

        api_model_names = {"GPT35":"gpt-3.5-turbo",
                           "GPT4":"gpt-4-1106-preview"}
        self.api_model_name = api_model_names[self.model]

    def get_skos_relation(self, concept1, concept2):
        """
        Obtains a SKOS relation prediction for a given concept pair using a GPT model.

        :param concept1: The first concept in the relation.
        :type concept1: str
        :param concept2: The second concept in the relation.
        :type concept2: str
        
        :return: The predicted relation and associated response text.
        :rtype: Tuple[str, str]
        """
        system_content, user_content = self.get_prompt(concept1, concept2)

        print(system_content)
        print(user_content)

        completion = client.chat.completions.create(
            model=self.api_model_name,
            temperature=TEMP,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
                ]
            )
        return completion.choices[0].message.content, "see pred column"

def main(args):
    """
    Main function to drive the prediction refinement process using specified model settings.

    :param args: Command-line arguments specifying the model, setting, and anchors.
    :type args: argparse.Namespace
    """
    global TEMP
    for TEMP in [0.0]:
        for test_set in ['stroma_g6_reference']:
            start_time = time.time()
            if "GPT" in args.model:
                classifier = GPTRefiner(test_set, args.model, args.setting, args.anchors)
            classifier.load_data()
            classifier.get_and_save_predictions()
            duration = time.time() - start_time
            print('Duration = ', duration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model", type=str, required=True, help="Choose which model was used? GPT35/GPT4?")
    parser.add_argument("--setting", type=str, required=True, help="Choose which setting to use, one-shot or few-shot")
    parser.add_argument("--anchors", type=str, required=True, help="Choose how many achors were used")
    args = parser.parse_args()

    main(args)