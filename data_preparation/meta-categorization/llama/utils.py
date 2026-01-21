import re
from typing import List
from numpy import random

rng = random.RandomState(42)

def parse_output(input_string: str) -> List[str]:
    input_list = input_string.split("\n")  # split the input by newline characters
    output_list = []
    for item in input_list:
        # find the first occurrence of a number in the item
        match = re.search(r"\d", item)
        if match:
            # add the item to the output list
            output_list.append(item)
    return output_list


def parse_output_number(input_string: str) -> List[str]:
    input_list = input_string.split("\n")  # split the input by newline characters
    output_list = []
    for item in input_list:
        # find the first occurrence of a number in the item
        match = re.search(r"\d", item)
        if match:
            # add the item to the output list
            output_list.append(item)
    return output_list

def parse_output_scores(input_string: str) -> List[str]:
    input_list = input_string.split("\n")  # split the input by newline characters
    output_list = []
    for item in input_list[1:]:
        item = item.lstrip(" "
        )  # remove any leading whitespace
        if item:  # skip empty items
            output_list.append(item)
    return output_list



def parse_output_list_scores(input_string: str) -> List[str]:
    input_list = input_string.split("\n")  # split the input by newline characters
    output_list = []
    for item in input_list[1:]:
        # split the item by whitespace
        if item:  # skip empty items
            # if it can be converted to a float, it is a score
            try:
                float(item)
                output_list.append(float(item))
            except ValueError:
                pass
    return output_list

def parse_output_list_binary(input_string: str) -> List[str]:
    input_list = input_string.split("\n")  # split the input by newline characters
    label_list = ["problem-solution", "call-to-action", "intention", "participation", "other"]
    output_list = []
    for item in input_list[1:]:
        output_list.append(item.lower().strip().replace("'", "").replace(".", ""))
    string_output = "".join(output_list)
    for label in label_list:
        if label in string_output.split(" "):
            return label
    return "other"