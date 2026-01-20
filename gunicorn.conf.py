from multiprocessing import cpu_count

bind: str = "0.0.0.0:8000"
workers: int = max(2, cpu_count() * 2 + 1)
worker_class: str = "uvicorn.workers.UvicornWorker"
timeout: int = 60
keepalive: int = 5
accesslog: str = "-"
errorlog: str = "-"
loglevel: str = "info"