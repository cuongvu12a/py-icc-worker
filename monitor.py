from multiprocessing import Process, Queue, Event
import time
import psutil
from datetime import datetime

def monitor_cpu_mem(stop_event, result_queue, poll_interval=0.05):
    cpu_data = []
    mem_data = []
    while not stop_event.is_set():
        try:
            cpu_data.append(psutil.cpu_percent(interval=None))
            mem_data.append(psutil.virtual_memory().percent)
            time.sleep(poll_interval)
        except Exception:
            break
    # Try to return results without blocking the child process.
    # Use non-blocking put so the child won't hang if the parent isn't
    # reading the queue (avoids deadlocks on some platforms/start-methods).
    try:
        result_queue.put_nowait((cpu_data, mem_data))
    except Exception:
        # If putting fails for any reason, just ignore and exit.
        pass

def wrapper_monitor(only_time=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            q = None
            stop_event = None
            monitor_proc = None
            if not only_time:
                q = Queue()
                stop_event = Event()
                monitor_proc = Process(target=monitor_cpu_mem, args=(stop_event, q))
                monitor_proc.start()

            result = func(*args, **kwargs)

            if not only_time:
                stop_event.set()  # signal monitor dừng
                monitor_proc.join(timeout=5)
                if monitor_proc.is_alive():
                    monitor_proc.terminate()

                cpu_data = []
                mem_data = []
                while True:
                    try:
                        data = q.get_nowait()
                        cpu_data, mem_data = data
                    except:
                        break

                # Close the queue and wait for the feeder thread to finish
                # so we don't leave background threads that can keep the
                # process alive on some platforms.
                try:
                    q.close()
                except Exception:
                    pass
                try:
                    q.join_thread()
                except Exception:
                    pass

                avg_cpu = sum(cpu_data)/len(cpu_data) if cpu_data else 0
                avg_mem = sum(mem_data)/len(mem_data) if mem_data else 0
                max_cpu = max(cpu_data) if cpu_data else 0
                max_mem = max(mem_data) if mem_data else 0

            end = datetime.now()
            duration = end - start
            print(f"Function {func.__name__} took {duration}")
            if not only_time:
                print(f"Average CPU usage: {avg_cpu:.2f}%")
                print(f"Average Memory usage: {avg_mem:.2f}%")
                print(f"Max CPU usage: {max_cpu:.2f}%")
                print(f"Max Memory usage: {max_mem:.2f}%")
            return result
        return wrapper
    return decorator
