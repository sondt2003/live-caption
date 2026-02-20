from abc import ABC, abstractmethod

class BaseTTS(ABC):
    @abstractmethod
    def generate(self, text: str, output_path: str, **kwargs) -> None:
        raise NotImplementedError

    def generate_batch(self, tasks: list) -> None:
        """
        Tasks is a list of dicts: [{"text": "...", "output_path": "...", ...}]
        """
        # Default implementation calls generate for each task (sub-optimal)
        for task in tasks:
            text = task.pop("text")
            output_path = task.pop("output_path")
            self.generate(text, output_path, **task)
