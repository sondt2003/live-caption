from abc import ABC, abstractmethod

class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, messages: list) -> str:
        """
        Base method for translation.
        Messages should be in OpenAI chat format: [{"role": "user", "content": "..."}]
        """
        pass
