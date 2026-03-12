from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict, Any, Optional, TypedDict, Unpack


@dataclass(frozen=True)
class GenerationConfig:
    backend: str = "openai"
    base_url: Optional[str] = None
    model: str = "gpt-5-nano"
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.05
    reasoning_effort: str = "minimal"
    reasoning_summary: Optional[str] = None
    verbosity: str = "low"
    max_retries: int = 3
    retry_delay: float = 0.5
    completion_window: str = "24h"
    poll_interval: float = 60.0
    timeout: Optional[float] = None

class GenerationOverrides(TypedDict, total=False):
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    repetition_penalty: float
    reasoning_effort: str
    reasoning_summary: Optional[str]
    verbosity: str
    max_retries: int
    retry_delay: float
    completion_window: str
    poll_interval: float
    timeout: Optional[float]

class BaseLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, schema: Optional[type], cfg: Optional[GenerationConfig], **overrides: Unpack[GenerationOverrides]) -> Union[str, Dict[str, Any]]:
        ...
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], cfg: Optional[GenerationConfig], custom_ids: Optional[List[str]], metadata: Optional[Dict[str, str]], **overrides: Unpack[GenerationOverrides]) -> List[Union[str, Dict[str, Any]]]:
        ...
    
    @abstractmethod
    async def async_generate(self, prompts: List[str], schema: Optional[type], cfg: Optional[GenerationConfig], concurrency: int, return_exceptions: bool, **overrides: Unpack[GenerationOverrides]) -> List[Union[str, Dict[str, Any]]]:
        ...