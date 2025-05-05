import asyncio


async def func1(job):
    """Simulates a computation-heavy task"""
    await asyncio.sleep(2)  # Simulating delay
    print(f"func1 done for {job}")
    return f"data_for_func4_{job}"


async def func2(job):
    """Another independent task"""
    await asyncio.sleep(3)
    print(f"func2 done for {job}")


async def func3(job):
    """Another independent task"""
    await asyncio.sleep(1)
    print(f"func3 done for {job}")


async def func4(job, data_from_func1):
    """Depends on func1's result"""
    await asyncio.sleep(2)
    print(f"func4 done for {job} with {data_from_func1}")


async def process_job(job):
    """Handles one job, orchestrating its functions"""

    # Run func1 first and wait for its result
    data_for_func4 = await func1(job)

    # Run func2, func3, and func4 in parallel
    task2 = asyncio.create_task(func2(job))
    task3 = asyncio.create_task(func3(job))
    task4 = asyncio.create_task(func4(job, data_for_func4))

    # Wait for all parallel tasks to complete
    await asyncio.gather(task2, task3, task4)


async def main():
    jobs = ["job1", "job2", "job3"]  # List of jobs

    # Process all jobs in parallel
    await asyncio.gather(*(process_job(job) for job in jobs))

# Run the event loop
asyncio.run(main())
