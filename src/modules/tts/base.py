from abc import ABC, abstractmethod

class BaseTTS(ABC):
    @abstractmethod
    def generate(self, text: str, output_path: str, **kwargs) -> None:
        """
        Base method for TTS generation.
        """
        pass
