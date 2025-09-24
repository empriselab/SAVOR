import os
import sys
import yaml
from PIL import Image
import numpy as np
import re
from functools import reduce
from typing import Any, Callable, Optional, Union, List
import torch

# GPT
import openai
import logging


class BaseLLMDriver:
    def __init__(self):
        self.device = torch.device('cuda')

    def reset(self):
        raise NotImplementedError("Subclasses to implement this method")
    
    def send_query(self, prompt: str, required_samples: int):
        raise NotImplementedError("Subclasses to implement this method")
    
######## Driver classes that directly run models ########

class ModelLLMDriver_GPT(BaseLLMDriver):
    def __init__(
        self,
        key_path="configs/openai_api_key.yaml",
        config_path="configs/gpt_config.yaml",
    ):
        super().__init__()
        with open(key_path, 'r') as f:
            key_dict = yaml.safe_load(f)
            api_key = key_dict['api_key']
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client = openai
        self.client.api_key = api_key
        print("Set up LLM driver!")

    def send_query(self, prompt, required_samples):
        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=prompt,
            n=required_samples,
            seed=self.config["seed"],
            temperature=self.config["temperature"]
        )
        return response
    

######## Model interface classes ########

class LLMInterface:
    def __init__(
        self,
        driver,
    ):
        self.driver = driver

    def reset(self):
        """
        Resets state of the planner, including clearing LLM context
        """
        raise NotImplementedError("Subclasses to implement this method")
    
    # TODO: Implement a batched version of query
    def query(
        self,
        prompt: str,
        validate_fn: Callable[[List[Any]], List[Any]],
        required_samples: int = 1,
        max_tries: int = 3,
        get_raw_responses: bool = False,
    ):
        answers = []
        raw_answers = []
        valid = False
        remaining_samples_needed = required_samples
        for _ in range(max_tries):
            response = self.driver.send_query(prompt, remaining_samples_needed)

            for choice in response.choices:
                validated_resp = validate_fn(choice.message.content)
                if validated_resp is not None:
                    answers.append(validated_resp)
                if get_raw_responses:
                    raw_answers.append(choice.message.content)

            remaining_samples_needed = required_samples - len(answers)
            valid = remaining_samples_needed <= 0
            if valid:
                break

        if get_raw_responses:
            return valid, answers, raw_answers
        else:
            return valid, answers
