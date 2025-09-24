"""
String utilities for formatting queries and responses to/from foundation models
"""
import base64
import re
import numpy as np

def generic_string_format(s: str):
    return (
        s.lower().replace(' ', '').replace('.', '')
    )

def extract_food_name(path, with_adj = False):
    # Find the index of the word "subject"
    subject_index = path.find("subject")
    
    if subject_index == -1:
        return "The word 'subject' was not found in the string."
    
    # Find the first underscore after "subject"
    first_underscore = path.find("_", subject_index)
    if first_underscore == -1:
        return "No underscores found after 'subject'."
    
    # Find the second underscore after the first underscore
    second_underscore = path.find("_", first_underscore + 1)
    if second_underscore == -1:
        return "Only one underscore found after 'subject'."
    foodname = path[first_underscore+1:second_underscore]
    if with_adj:
        return foodname
    foodname = re.sub(r"\(.*?\)", "", foodname)
    return foodname

def extract_food_nam_from_bagpath(path):
    # Split the path into parts by '/'
    parts = path.split("/")
    
    # Find the part that contains "_skewer"
    for part in parts:
        if "_skewer" in part or "_scoop" in part or "_dip" in part:
            # Extract the food name by splitting at '_'
            return part.split("_")[0]
    return None


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

def sampling(sorted_list, n):
    if n <= 0 or n > len(sorted_list):
        raise ValueError("n must be between 1 and the length of the list.")
    indices = np.linspace(0, len(sorted_list) - 1, n, dtype=int)
    return [sorted_list[i] for i in indices]

