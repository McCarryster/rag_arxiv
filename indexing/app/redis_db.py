import redis
from typing import Dict, Any, cast

class RedisManager:
    def __init__(self, RD: Dict[str, Any]) -> None:
        """
        Initializes the Redis client using a configuration dictionary.
        
        Args:
            RD: Dictionary containing connection parameters (host, port, db, etc.)
        """
        self.redis_config: Dict[str, Any] = RD
        self.client: redis.Redis = redis.Redis(
            **self.redis_config,
            decode_responses=True
        )

    def is_duplicate(self, name: str, file_hash: str) -> bool:
        """
        Check if hash exists in the Redis set.
        
        Args:
            name: The key name of the Redis set.
            file_hash: The SHA-256 string to check.
        """
        return self.client.sismember(name, file_hash) == 1

    def add_hash(self, name: str, file_hash: str) -> None:
        """
        Add hash to the Redis set.
        
        Args:
            name: The key name of the Redis set.
            file_hash: The SHA-256 string to add.
        """
        self.client.sadd(name, file_hash)
    
    def get_count_hashes(self, name: str) -> int:
        """
        Returns the total number of unique hashes stored in the specified set.
        
        Args:
            name: The key name of the Redis set.
            
        Returns:
            int: The cardinality (count) of the set. Returns 0 if the set doesn't exist.
        """
        count = cast(int, self.client.scard(name))
        return count
