from pydantic import BaseModel

class ParseError(Exception):
    pass

class Parser:
    def __init__(self, protocol: type[BaseModel], type: str = "json"):
        self.protocol = protocol
        self.type = type
    
    def parse(self, text: str) -> BaseModel:
        if self.type == "json":
            try:
                left_brace = text.find("{")
                right_brace = text.rfind("}")
                if left_brace == -1 or right_brace == -1:
                    raise ParseError(f"Failed to find JSON: {text}")
                json_text = text[left_brace:right_brace+1]
                return self.protocol.model_validate_json(json_text).model_dump()
            except Exception as e:
                raise ParseError(f"Failed to parse JSON: {text}, error: {e}")