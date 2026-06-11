import os
import redis

_uri = os.environ.get("REDIS_URI", "")

# redis-py connection pools are lazy - no actual TCP connection is made here.
# Commands will fail at call-time if Redis is not reachable.
client = redis.Redis.from_url(_uri, decode_responses=True) if _uri else None
