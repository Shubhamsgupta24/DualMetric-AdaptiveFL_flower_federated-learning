import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from tabulate import tabulate
import sys

NUM_CLIENTS = 2

def run_client(client_id):
    try:
        process = subprocess.Popen(
            ["python", "client.py", str(client_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"Error starting client {client_id}: {e}")
        return None

def monitor_client(client_id, process):
    if process is None:
        return [(client_id, "Failed to start")]
    outputs = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            outputs.append((client_id, output.strip()))
    return outputs

def main():
    processes = []
    with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
        for i in range(NUM_CLIENTS):
            process = run_client(i)
            processes.append(process)
            executor.submit(monitor_client, i, process)

    all_outputs = []
    for i, process in enumerate(processes):
        outputs = monitor_client(i, process)
        all_outputs.extend(outputs)

    # Sort outputs by client ID
    all_outputs.sort(key=lambda x: x[0])

    # Display outputs in tabular format
    if all_outputs:
        table = tabulate(all_outputs, headers=["Client ID", "Output"], tablefmt="grid")
        print(table)
    else:
        print("No output from clients")

if __name__ == "__main__":
    main()
