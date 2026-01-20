import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
import sys
import tqdm

# append a new directory to sys.path
sys.path.append('/Users/arpe/Documents/ITU/Projects/formafluens/meta-categorization/llama/')

from config import HF_HUB_MODELS
from prompting.datasets_config import (
    AssignConcretenessScore, AssignAuditoryScore, AssignGustatoryScore, AssignHapticScore, AssignInteroceptiveScore, AssignOlfactoryScore, AssignVisualScore,
    AssignFootLegScore, AssignHandArmScore, AssignHeadScore, AssignMouthScore, AssignTorsoScore, MetaCategorization, AssignImagineryScore
)

from utils import parse_output

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

task = "imaginery" # "concreteness", "auditory", "gustatory", "haptic", "interoceptive", "olfactory", "visual", "footleg", "handarm", "head", "mouth", "torso", "meta"

if task == "concreteness":
    list_words = pd.read_csv("../../missing_categories_concreteness.csv")

elif task == "meta":
    list_words = pd.read_csv("../../all_categories.csv", names=["category"], header=None)

elif task == "imaginery":
    list_words = pd.read_csv("../../missing_categories_imageability.csv")

else:
    list_words = pd.read_csv(f"../../missing_categories_sensorimotor.csv")

dictionary_tasks = {"concreteness": AssignConcretenessScore,
                    "auditory": AssignAuditoryScore,
                    "gustatory": AssignGustatoryScore,
                    "haptic": AssignHapticScore,
                    "interoceptive": AssignInteroceptiveScore,
                    "olfactory": AssignOlfactoryScore,
                    "visual": AssignVisualScore,
                    "footleg": AssignFootLegScore,
                    "handarm": AssignHandArmScore,
                    "head": AssignHeadScore,
                    "mouth": AssignMouthScore,
                    "torso": AssignTorsoScore,
                    "meta": MetaCategorization,
                    "imaginery": AssignImagineryScore,
                    }

class HuggingfaceChatTemplate:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.use_default_system_prompt = False

    def get_template_classification(self, system_prompt: str, task: str) -> str:
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\nWord: {text}\nScore: """.format(
                    task=task,
                    text="{text}",
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(chat, tokenize=False)
    
    def get_template_metacategorization(self, system_prompt: str, task: str) -> str:
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\nWords: {text}\Answer: """.format(
                    task=task,
                    text="{text}",
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(chat, tokenize=False)

if __name__ == "__main__":

    if task != "meta":

        outputs = []

        for index, row in tqdm.tqdm(list_words.iterrows(), total=list_words.shape[0]):

            config = dictionary_tasks[task]

            llm = InferenceClient(
                model=HF_HUB_MODELS["llama-3.3-70b"],
                token="HF token here",
            )

            template = HuggingfaceChatTemplate(
                model_name=HF_HUB_MODELS["llama-3.3-70b"]
            ).get_template_classification(
                system_prompt=config.classification_system_prompt,
                task=config.classification_task_prompt,
            )

            output = llm.text_generation(
                template.format(
                    text=row["category"].replace("_", " ")
                ),
                max_new_tokens=20,
                temperature=0.1,
                )

            # Parse output
            parsed_output = parse_output(output)

            print(row["category"])
            print(parsed_output)

            outputs.append([row["category"], parsed_output])

    else:

        outputs = []

        list_words = list_words["category"].tolist()


        config = dictionary_tasks[task]

        llm = InferenceClient(
            model=HF_HUB_MODELS["llama-3.3-70b"],
            token="HF token here",
        )

        template = HuggingfaceChatTemplate(
            model_name=HF_HUB_MODELS["llama-3.3-70b"]
        ).get_template_metacategorization(
            system_prompt=config.classification_system_prompt,
            task=config.classification_task_prompt,
        )

        output = llm.text_generation(
            template.format(
                text=",".join(list_words)
            ),
            max_new_tokens=4000,
            temperature=0.1,
            )

        print(output)
        
        sys.exit(0)


    # Save outputs to df and csv
    df_outputs = pd.DataFrame(outputs, columns=["category", "score"])

    df_outputs.to_csv(f"../results/{task}_scores_llama3.3.csv", index=False)