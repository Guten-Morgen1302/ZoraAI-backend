from __future__ import annotations

import logging
import os
import time
import importlib
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("zora.middleware.rate_limiter")


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass
class RateLimitDecision:
    allowed: bool
    remaining_tokens: int
    retry_after_seconds: int
    enforced: bool
    capacity: int


class RedisTokenBucketLimiter:
    """Redis-backed token bucket limiter with fail-open behavior."""

    LUA_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now_ts = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])
local ttl_seconds = tonumber(ARGV[5])

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
end
if ts == nil then
    ts = now_ts
end

local elapsed = now_ts - ts
if elapsed < 0 then
    elapsed = 0
end

local refilled = tokens + (elapsed * refill_rate)
if refilled > capacity then
    refilled = capacity
end

local allowed = 0
local updated_tokens = refilled
if refilled >= requested then
    allowed = 1
    updated_tokens = refilled - requested
end

redis.call('HMSET', key, 'tokens', updated_tokens, 'ts', now_ts)
redis.call('EXPIRE', key, ttl_seconds)

return {allowed, updated_tokens}
""".strip()

    def __init__(
        self,
        redis_client: Any,
        capacity: int,
        refill_rate: float,
        key_prefix: str,
        bucket_ttl_seconds: int,
        enabled: bool,
    ):
        self.redis_client = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.key_prefix = key_prefix
        self.bucket_ttl_seconds = bucket_ttl_seconds
        self.enabled = enabled

    @classmethod
    def from_env(cls) -> "RedisTokenBucketLimiter":
        enabled = _as_bool(os.getenv("RATE_LIMIT_ENABLED"), True)
        capacity = max(1, int(float(os.getenv("RATE_LIMIT_CAPACITY", "60"))))
        refill_rate = max(0.01, float(os.getenv("RATE_LIMIT_REFILL_RATE", "1.0")))
        key_prefix = os.getenv("RATE_LIMIT_KEY_PREFIX", "rate_limit:token_bucket")
        default_ttl = max(60, int((capacity / refill_rate) * 2))
        bucket_ttl_seconds = max(1, int(float(os.getenv("RATE_LIMIT_BUCKET_TTL_SECONDS", str(default_ttl)))))

        redis_client = None
        if enabled:
            redis_client = cls._build_redis_client()
            if redis_client is None:
                enabled = False

        return cls(
            redis_client=redis_client,
            capacity=capacity,
            refill_rate=refill_rate,
            key_prefix=key_prefix,
            bucket_ttl_seconds=bucket_ttl_seconds,
            enabled=enabled,
        )

    @staticmethod
    def _build_redis_client() -> Any | None:
        try:
            redis = importlib.import_module("redis")
        except ImportError:
            logger.warning("redis package not installed; rate limiting disabled")
            return None

        redis_url = os.getenv("REDIS_URL")
        redis_tls = _as_bool(os.getenv("REDIS_TLS"), False)

        try:
            if redis_url:
                kwargs: dict[str, Any] = {"decode_responses": True}
                if redis_url.startswith("redis://") and redis_tls:
                    kwargs["ssl"] = True
                client = redis.Redis.from_url(redis_url, **kwargs)
            else:
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
                username = os.getenv("REDIS_USERNAME")
                password = os.getenv("REDIS_PASSWORD")
                client = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    username=username,
                    password=password,
                    ssl=redis_tls,
                )

            client.ping()
            logger.info("Redis rate limiter initialized")
            return client
        except Exception as exc:  # noqa: BLE001 - startup should not fail when Redis is unavailable
            logger.warning("Unable to initialize Redis rate limiter: %s", exc)
            return None

    def consume(self, identifier: str, requested_tokens: int = 1) -> RateLimitDecision:
        if not self.enabled or self.redis_client is None:
            return RateLimitDecision(
                allowed=True,
                remaining_tokens=self.capacity,
                retry_after_seconds=0,
                enforced=False,
                capacity=self.capacity,
            )

        bucket_key = f"{self.key_prefix}:{identifier}"
        now_ts = time.time()

        try:
            response = self.redis_client.eval(
                self.LUA_SCRIPT,
                1,
                bucket_key,
                self.capacity,
                self.refill_rate,
                now_ts,
                max(1, requested_tokens),
                self.bucket_ttl_seconds,
            )

            allowed = bool(int(response[0]))
            remaining_float = float(response[1]) if len(response) > 1 else 0.0
            remaining = max(0, int(remaining_float))

            retry_after = 0
            if not allowed:
                missing_tokens = max(0.0, float(requested_tokens) - remaining_float)
                retry_after = max(1, int(missing_tokens / self.refill_rate))

            return RateLimitDecision(
                allowed=allowed,
                remaining_tokens=remaining,
                retry_after_seconds=retry_after,
                enforced=True,
                capacity=self.capacity,
            )
        except Exception as exc:  # noqa: BLE001 - fail open to preserve availability
            logger.warning("Redis rate limiter failed during request: %s", exc)
            return RateLimitDecision(
                allowed=True,
                remaining_tokens=self.capacity,
                retry_after_seconds=0,
                enforced=False,
                capacity=self.capacity,
            )
