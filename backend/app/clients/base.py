import time
import random
import logging
from typing import Callable, Any, Tuple, Type, Optional
from functools import wraps
import requests

logger = logging.getLogger("bel_chatbot.clients")

DEFAULT_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    requests.RequestException,
    requests.Timeout,
    requests.ConnectionError,
)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 5.0,
    retryable_status_codes: Tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
):
    """
    Decorator for resilient HTTP client execution with exponential backoff and jitter.

    Handles transient rate-limiting (429), gateway/server errors (500, 502, 503, 504),
    and network connection / timeout failures gracefully.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_retries + 1):
                try:
                    res = func(*args, **kwargs)
                    if isinstance(res, requests.Response):
                        if res.status_code in retryable_status_codes:
                            if attempt == max_retries:
                                logger.warning(
                                    f"[{func.__name__}] Retry limit ({max_retries}) reached with HTTP {res.status_code}."
                                )
                                return res
                            jitter = random.uniform(0.8, 1.2)
                            sleep_time = min(delay * jitter, max_delay)
                            logger.warning(
                                f"[{func.__name__}] Retryable HTTP {res.status_code} on attempt {attempt}/{max_retries}. "
                                f"Backing off for {sleep_time:.2f}s..."
                            )
                            time.sleep(sleep_time)
                            delay *= backoff_factor
                            continue
                    return res
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"[{func.__name__}] Exhausted all {max_retries} retry attempts. Final error: {e}"
                        )
                        raise e
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = min(delay * jitter, max_delay)
                    logger.warning(
                        f"[{func.__name__}] Network error on attempt {attempt}/{max_retries}: {e}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= backoff_factor

            if last_exception:
                raise last_exception
            return None

        return wrapper

    return decorator
