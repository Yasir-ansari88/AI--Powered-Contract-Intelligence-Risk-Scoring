import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_BASE = "http://127.0.0.1:8001"
NUM_REQUESTS = 20       
CONCURRENCY = 5         
PDF_PATH = "sample_contract.pdf"  


def hit_health():
    start = time.time()
    try:
        res = requests.get(f"{API_BASE}/health", timeout=10)
        return res.status_code == 200, time.time() - start
    except Exception:
        return False, time.time() - start


def hit_search():
    start = time.time()
    try:
        res = requests.get(f"{API_BASE}/search", params={"q": "either party may terminate without cause"}, timeout=30)
        return res.status_code == 200, time.time() - start
    except Exception:
        return False, time.time() - start


def run_load_test(label: str, fn, num_requests: int, concurrency: int):
    print(f"\n=== {label}: {num_requests} requests, concurrency={concurrency} ===")
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(fn) for _ in range(num_requests)]
        for future in as_completed(futures):
            results.append(future.result())

    successes = [r for ok, r in results if ok]
    failures = [r for ok, r in results if not ok]
    times = [r for _, r in results]

    print(f"Success: {len(successes)}/{num_requests}  |  Failed: {len(failures)}/{num_requests}")
    if times:
        print(f"Avg: {statistics.mean(times):.3f}s  |  Min: {min(times):.3f}s  |  Max: {max(times):.3f}s")
        if len(times) >= 2:
            print(f"Median: {statistics.median(times):.3f}s  |  Std dev: {statistics.stdev(times):.3f}s")


if __name__ == "__main__":
    run_load_test("Health check endpoint", hit_health, NUM_REQUESTS, CONCURRENCY)
    run_load_test("Semantic search endpoint", hit_search, 10, 3)

    print("\nDone. For upload-endpoint load testing, run this manually a few times")
    print("via the Swagger UI or frontend — concurrent heavy PDF uploads on a")
    print("single-process local server will naturally queue up; that's expected")
    print("behavior to note in the final report, not a bug.")
