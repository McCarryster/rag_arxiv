import functools
import logging
import traceback

from langfuse import get_client
import config


logger = logging.getLogger(__name__)


def async_trace(name: str, model: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*func_args, **func_kwargs):
            if not config.LANGFUSE_AVAILABLE:
                return await func(*func_args, **func_kwargs)
            
            langfuse = get_client()

            try:
                with langfuse.start_as_current_observation(as_type="generation", name=name, model=model) as generation:
                    # Execute the function
                    result = await func(*func_args, **func_kwargs)
                    generation.update(
                        input=result['input'],
                        output=result['output'],
                        usage_details={
                            "input": result['input_tokens'],
                            "total": result['total_tokens']
                        },
                        metadata=result.get('metadata', {})
                    )
                return result
            except BaseException as be:
                err_str = traceback.format_exc()
                logger.error(f"Exception in traced function {name}: {be}, Trace: {err_str}")
                langfuse.flush()
                raise

        return wrapper
    return decorator