import re
import yaml
from typing import Any, Optional, Union, Callable, List, Tuple, Type, Dict
from functools import reduce
import base64

from utils.string_utils import generic_string_format

################ Prompt registry and abstract class ################

class BasePrompt:
    def __init__(self, template):
        self.template = template

    def generatePrompt(self, ctx=None):
        raise NotImplementedError("Subclasses should implement this method")
    
    def generateHandler(self, resp: Any, ctx=None):
        raise NotImplementedError("Subclasses should implement this method")
        
class PromptRegistry:
    _registry = {}
    with open('utils/template_prompts.yaml') as f:
        _prompt_templates = yaml.safe_load(f)

    def __init__(self):
        self.log = None

    @classmethod
    def register(cls, prompt_class: Type[BasePrompt]):
        name = prompt_class.__name__
        if name in cls._registry:
            raise ValueError(f'Class {name} is already registered!')
        cls._registry[name] = prompt_class

    def getPromptAndHandler(
        self, 
        prompt_class: Union[str, Type[BasePrompt]],
        ctx=None,
    ) -> Tuple[Any, Callable]:
        """
        Input: prompt_class, i.e. type of prompt specified as str or as class
               ctx, contextual/scene info needed to populate the prompt,
        Output: prompt, string or list, depending on the prompt type
                handler_fn, function to validate and format the response,
                            may be curried with context if that is needed
        """
        if not isinstance(prompt_class, str):
            prompt_class = prompt_class.__name__
        if prompt_class not in self._registry:
            raise ValueError(f'Request prompt {prompt_class} not registered!')
        
        prompt_object = self._registry[prompt_class](
            self._prompt_templates[prompt_class]
        )
        prompt = prompt_object.generatePrompt(ctx)
        handler = lambda r: prompt_object.generateHandler(r, ctx)
        return prompt, handler

def register_prompt(prompt_class):
    PromptRegistry.register(prompt_class)
    return prompt_class

################ Actual prompt implementations ################

class Prompts: # "Prompts" namespace in which to implement prompts

    @register_prompt
    class ActionSelection(BasePrompt):
        def generatePrompt(self, ctx: Dict[str, str]) -> List[Dict[str, str]]:
            """
            Input: ctx, dict containing descriptions of calibration summary, attempt history, and current food context.
            """

            context = self.template['ctx']
            # query = self.template['query'].format(
            #     food_name=ctx['food_name'],
            #     Softness=ctx['Softness'],
            #     Moisture=ctx['Moisture'],
            #     Viscosity=ctx['Viscosity']
            # )
            query = self.template['query_w_instruction'].format(
                food_name=ctx['food_name'],
                Shape=ctx['Shape'],
                Size=ctx['Size'],
                Softness=ctx['Softness'],
                Moisture=ctx['Moisture'],
                Viscosity=ctx['Viscosity'],
                CalibrationSummary=ctx['CalibrationSummary'],
                AttemptHistory=ctx['AttemptHistory']
                )
            fewshot = self.template['fewshot']
            chat = [
                {"role": "system", "content": context}
            ] + [
                {"role": k.split('_')[0], "content": v} for k, v in fewshot.items()
            ] + [
                {"role": "user", "content": query}
            ]
            return chat
        
        def generateHandler(self, resp: str, ctx=None) -> Optional[bool]:
            """
            Input: resp, string response from LLM
            Output: option, where None indicates an invalid response,
                    otherwise a boolean indicating matching validity
            """
            answer = resp.split("Answer: ")[-1].lower()
            print(resp)
            if len(answer) != 0:
                return answer
            else:
                return None
  
    @register_prompt
    class ImageQuery(BasePrompt):
        def generatePrompt(self, ctx: Dict[str, str]) -> List[Dict[str, str]]:
            """
            Input: ctx, dict containing descriptions of observed and candidate places
            """

            # TODO: Currently does not handle specs with more than one place class
            context = self.template['ctx']
            query = self.template['query'].format(
                food_name=ctx['food_name'],
                AttemptHistory=ctx['AttemptHistory']
            )
            base64_image = ctx['base64_image']
            fewshot = self.template['fewshot']
            chat = [
                {"role": "system", "content": context}
            ] + [
                {"role": k.split('_')[0], "content": v} for k, v in fewshot.items()
            ] + [
                {"role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                        },
                    },
                ],}
            ]
            return chat
        
        def generateHandler(self, resp: str, ctx=None) -> Optional[bool]:
            """
            Input: resp, string response from LLM
            Output: option, where None indicates an invalid response,
                    otherwise a boolean indicating matching validity
            """
            answer = resp.split("Answer:")[-1].lower()
            split_parts = re.split(r'[:;.]', answer)
            print(split_parts)
            scores = [float(part.strip()) for part in split_parts if part.strip().isdigit()]
            for i in range(len(split_parts)):
                if 'shape' in split_parts[i] and i+1 < len(split_parts):
                    scores.append( split_parts[i+1])
                elif 'size' in split_parts[i] and 'sized' not in split_parts[i] and i+1 < len(split_parts):
                    scores.append( split_parts[i+1])
            print(scores)
            if len(scores) == 5:
                return scores
            else:
                return None
  