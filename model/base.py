from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any


class BaseLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Union[str, Dict[str, Any]]:
        ...
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Union[str, Dict[str, Any]]]:
        ...
    
    @abstractmethod
    async def async_generate(self, prompts: List[str], **kwargs) -> List[Union[str, Dict[str, Any]]]:
        ...