import ray
import time
import random

ray.init(ignore_reinit_error=True)  # Initialize Ray


@ray.remote
def func1(job):
    """Simulates a computation-heavy task with random execution time"""
    delay = random.uniform(1, 3)  # Random delay between 1-3 seconds
    time.sleep(delay)
    print(f"func1 done for {job} (time: {delay:.2f}s)")
    return f"data_for_func4_{job}"


@ray.remote
def func2(job):
    """Another independent task with random execution time"""
    delay = random.uniform(2, 5)
    time.sleep(delay)
    print(f"func2 done for {job} (time: {delay:.2f}s)")


@ray.remote
def func3(job):
    """Another independent task with random execution time"""
    delay = random.uniform(1, 4)
    time.sleep(delay)
    print(f"func3 done for {job} (time: {delay:.2f}s)")


@ray.remote
def func4(job, data_from_func1):
    """Depends on func1's result with random execution time"""
    delay = random.uniform(1, 3)
    time.sleep(delay)
    print(f"func4 done for {job} with {data_from_func1} (time: {delay:.2f}s)")


def process_job(job):
    """Handles one job, orchestrating its functions"""

    # Start func1 first
    future_func1 = func1.remote(job)

    # Start func2 and func3 in parallel
    future_func2 = func2.remote(job)
    future_func3 = func3.remote(job)

    # Start func4 only after func1 completes
    future_func4 = func4.remote(job, ray.get(future_func1))

    # Wait for func2
    #


def main():
    jobs = ["job1", "job2", "job3"]  # List of jobs

    # Launch all jobs in parallel using Ray
    ray.get([ray.remote(process_job).remote(job) for job in jobs])


if __name__ == "__main__":
    main()
    ray.shutdown()  # Cleanup Ray
