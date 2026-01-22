import functools
import logging
import traceback
from typing import Callable, Optional

import asyncio
import inspect
from langfuse import get_client
import config


logger = logging.getLogger(__name__)


def get_parent_context(func):
    """
    Determines the parent class name (if method) or module name (if function).
    Designed to work inside decorators.
    """
    # 1. Unwrap if the decorator is placed *after* @classmethod/@staticmethod
    #    e.g. @trace @classmethod def foo...
    if isinstance(func, (classmethod, staticmethod)):
        func = func.__func__

    # 2. Try to get the Class Name from qualname
    #    qualname is usually "ClassName.method_name"
    if hasattr(func, '__qualname__'):
        qualname = func.__qualname__
        if '.' in qualname and '<locals>' not in qualname:
            # Split "MyClass.my_method" -> returns "MyClass"
            return qualname.rsplit('.', 1)[0]

    # 3. Fallback: Standalone function -> Return Module Name
    if hasattr(func, '__module__') and func.__module__:
        return func.__module__

    # 4. Deep Fallback: Get Filename from code object
    #    Useful if module is unresolvable
    if hasattr(func, '__code__'):
        return inspect.getfile(func)

    return "unknown"


# def async_trace(
#     name: Optional[str] = None,
#     name_extractor: Optional[Callable] = None,
# ):
#     """
#     Decorator for tracing functions that are called within process_signal.

#     This decorator:
#     - Inherits context from parent trace (initialized in process_signal)
#     - Creates a child span using context managers
#     - Names observations

#     Usage:
#         @trace()
#         @track_activity("Processing ...")
#         def process_stuff(self, ...):
#             pass
#     """

#     def decorator(func):
#         try:
#             parent_class_name = get_parent_context(func=func)
#             func_name = func.__name__
#         except Exception as e:
#             logger.error(f"Error in trace decorator setup: {e}")
#             parent_class_name = "unknown"
#             func_name = "unknown"

#         @functools.wraps(func)
#         async def wrapper(*func_args, **func_kwargs):
#             if not config.LANGFUSE_AVAILABLE:
#                 return func(*func_args, **func_kwargs)
            
#             langfuse = get_client()
#             observation_name = f"{parent_class_name}.{func_name}" if name is None else name

#             if name_extractor:
#                 extracted_name = name_extractor(*func_args, **func_kwargs)
#                 observation_name = extracted_name or observation_name

#             # Create a child span - automatically becomes child of active observation
#             try:
#                 with langfuse.start_as_current_observation(as_type="span", name=observation_name) as span:
#                     # Execute the function
#                     result = await func(*func_args, **func_kwargs)  # Await async function
#                     span.update(input={"args": func_args, "kwargs": func_kwargs}, output=result)
#                 return result

#             except BaseException as be:
#                 err_str = traceback.format_exc()
#                 logger.error(f"Exception in traced function {observation_name}: {be}, Trace: {err_str}")
#                 langfuse.flush()
#                 raise

#         return wrapper
#     return decorator


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