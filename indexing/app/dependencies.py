from typing import Optional
from redis_db import RedisManager
import config

_redis_manager: Optional[RedisManager] = None
_redis_manager: Optional[RedisManager] = None

def get_redis_manager(recreate: bool = False) -> RedisManager:
    """
    Get or create the singleton RedisManager instance.
    
    Args:
        recreate: Force creation of a new instance even if one exists
        
    Returns:
        RedisManager instance
    """
    global _redis_manager
    # Only create if it doesn't exist OR if the path has changed (important for tests!)
    current_redis_config = config.RD
    
    if _redis_manager is None or _redis_manager.redis_config != current_redis_config or recreate:
        _redis_manager = RedisManager(current_redis_config)
    
    return _redis_manager

