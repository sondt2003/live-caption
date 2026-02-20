import time
import json
import os
from loguru import logger

class PerformanceTracker:
    def __init__(self):
        self.stages = {}
        self.start_time = time.time()
        self.total_duration = 0
        self._current_stage = None

    def start_stage(self, name):
        """Bắt đầu tính thời gian cho một giai đoạn cụ thể."""
        if self._current_stage:
            self.end_stage(self._current_stage)
        
        self._current_stage = name
        self.stages[name] = {
            "start": time.time(),
            "end": None,
            "duration": None
        }
        logger.debug(f"[PERF] Starting stage: {name}")

    def end_stage(self, name=None):
        """Kết thúc tính thời gian cho một giai đoạn cụ thể."""
        target_name = name or self._current_stage
        if not target_name or target_name not in self.stages:
            return

        end_time = time.time()
        self.stages[target_name]["end"] = end_time
        duration = end_time - self.stages[target_name]["start"]
        self.stages[target_name]["duration"] = round(duration, 3)
        
        if target_name == self._current_stage:
            self._current_stage = None
        
        logger.debug(f"[PERF] Finished stage: {target_name} ({round(duration, 3)}s)")

    def finalize(self):
        """Tính tổng thời gian và hoàn tất."""
        if self._current_stage:
            self.end_stage(self._current_stage)
        
        self.total_duration = round(time.time() - self.start_time, 3)
        return self.get_stats()

    def get_stats(self):
        """Trả về các thống kê đã thu thập."""
        stats = {
            "total_duration": self.total_duration,
            "stages": [
                {
                    "name": name,
                    "duration": info["duration"]
                } for name, info in self.stages.items()
            ]
        }
        return stats

    def save_stats(self, file_path):
        """Lưu thống kê vào file JSON."""
        stats = self.get_stats()
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
            logger.info(f"[PERF] Timing stats saved to {file_path}")
        except Exception as e:
            logger.error(f"[PERF] Failed to save stats: {e}")
