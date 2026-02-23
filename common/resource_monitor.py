import os
import time
import psutil
import threading

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.log = []
        self.running = True

    def run(self):
        while self.running:
            # RSS is the Physical RAM (RES in top)
            ram_usage = self.process.memory_info().rss / (1024 ** 2) # MB
            cpu_percent = self.process.cpu_percent(interval=None)   # % across all threads
            self.log.append({
                "timestamp": time.time(),
                "ram_mb": ram_usage,
                "cpu_percent": cpu_percent
            })
            time.sleep(self.interval)

    def stop(self):
        self.running = False