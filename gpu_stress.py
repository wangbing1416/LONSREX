import torch
import torch.multiprocessing as mp
import time
import sys
import requests

def is_server_ready(url, timeout=5):
    """Check if the server is reachable and responds with 200"""
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False

def gpu_worker(device_index, stop_event):
    """Single GPU stress worker"""
    torch.cuda.set_device(device_index)

    matrix_size = 2048
    a = torch.rand((matrix_size, matrix_size), device=f'cuda:{device_index}', requires_grad=True)
    b = torch.rand((matrix_size, matrix_size), device=f'cuda:{device_index}', requires_grad=True)

    print(f"[Worker-{device_index}] GPU stress task started")
    while not stop_event.is_set():
        c = torch.matmul(a, b)
        loss = c.sum()
        loss.backward()
        time.sleep(0.00001)

    print(f"[Worker-{device_index}] Stop signal received, ending task")

def main(server_url, max_wait=300):
    num_gpus = torch.cuda.device_count()
    stop_event = mp.Event()
    processes = []

    # Launch stress task on each GPU
    for gpu_index in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(gpu_index, stop_event))
        p.start()
        processes.append(p)

    start_time = time.time()
    print(f"Running stress test and checking if server {server_url} is ready, up to {max_wait} seconds...")

    # Check readiness while stress tasks run
    while time.time() - start_time < max_wait:
        if is_server_ready(server_url):
            print(f"✅ Server {server_url} is reachable and responding")
            break
        time.sleep(1)

    # Stop stress tasks
    stop_event.set()
    for p in processes:
        p.join()

    if time.time() - start_time >= max_wait:
        print("⚠️ Timeout: server not ready within the limit")
    else:
        print("GPU stress tasks finished, proceed to the next step")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    server_url = "http://127.0.0.1:30011/get_model_info"

    if len(sys.argv) >= 2:
        max_wait = int(sys.argv[1])
    else:
        max_wait = 300

    main(server_url, max_wait)
