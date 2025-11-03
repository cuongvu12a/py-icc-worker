from multiprocessing import Process, Event
import time
import psutil
from datetime import datetime

def monitor_cpu_mem(stop_event, func_name, poll_interval=0.05, log_file=None):
    import psutil, time
    from datetime import datetime

    cpu_data_percent = []
    mem_data_percent = []
    start = datetime.now()

    total_mem_gb = psutil.virtual_memory().total / (1024**3)

    try:
        while not stop_event.is_set():
            cpu_percent = psutil.cpu_percent(interval=poll_interval)
            mem_info = psutil.virtual_memory()
            mem_percent = mem_info.percent

            cpu_data_percent.append(cpu_percent)
            mem_data_percent.append(mem_percent)
    except Exception:
        pass

    end = datetime.now()
    duration_min = (end - start).total_seconds() / 60

    avg_cpu_percent = sum(cpu_data_percent)/len(cpu_data_percent) if cpu_data_percent else 0
    avg_mem_percent = sum(mem_data_percent)/len(mem_data_percent) if mem_data_percent else 0
    max_cpu_percent = max(cpu_data_percent) if cpu_data_percent else 0
    max_mem_percent = max(mem_data_percent) if mem_data_percent else 0

    avg_mem_gb = avg_mem_percent / 100 * total_mem_gb
    max_mem_gb = max_mem_percent / 100 * total_mem_gb

    msg = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Function '{func_name}' executed in {duration_min:.2f} phút | "
        f"CPU avg={avg_cpu_percent:.1f}%, "
        f"max={max_cpu_percent:.1f}% | "
        f"RAM avg={avg_mem_percent:.1f}% ({avg_mem_gb:.2f}/{total_mem_gb:.1f}GB), "
        f"max={max_mem_percent:.1f}% ({max_mem_gb:.2f}/{total_mem_gb:.1f}GB)"
    )

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    else:
        print(msg)

def wrapper_monitor(only_time=False, log_file=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            stop_event = Event()
            monitor_proc = None

            if not only_time:
                monitor_proc = Process(
                    target=monitor_cpu_mem,
                    args=(stop_event, func.__name__),
                    kwargs={"poll_interval": 0.1, "log_file": log_file},
                    daemon=True,
                )
                monitor_proc.start()

            start = datetime.now()
            result = func(*args, **kwargs)
            end = datetime.now()

            if not only_time:
                stop_event.set()
                monitor_proc.join(timeout=2)
                if monitor_proc.is_alive():
                    monitor_proc.terminate()
            else:
                duration = (end - start).total_seconds() / 60
                print(f"Function '{func.__name__}' executed in {duration:.2f} phút")

            return result
        return wrapper
    return decorator
