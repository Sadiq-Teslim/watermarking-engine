"""RQ worker entrypoint. Listens on the `fpwm` queue.

Tasks are registered in later batches (worker/tasks.py). For now this starts a real
worker that connects to Redis and waits for jobs — no fake/no-op behaviour.
"""
import redis
from rq import Queue, Worker

from app.config import get_settings

QUEUE_NAME = "fpwm"


def main() -> None:
    settings = get_settings()
    connection = redis.Redis.from_url(settings.redis_url)
    worker = Worker([Queue(QUEUE_NAME, connection=connection)], connection=connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
