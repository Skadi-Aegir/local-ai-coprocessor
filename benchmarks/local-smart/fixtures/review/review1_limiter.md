"""REVIEW TARGET 1 - deliberately flawed patch. Do not "fix"; it is a fixture.

The diff below is proposed for a rate-limiter service. Review it.

```diff
--- a/limiter.py
+++ b/limiter.py
@@
 class TokenBucket:
-    def __init__(self, rate: float, burst: int):
-        self.rate = rate
-        self.burst = burst
-        self.tokens = burst
-        self.updated = time.monotonic()
+    def __init__(self, rate: float, burst: int):
+        self.rate = rate
+        self.burst = burst
+        self.tokens = burst
+        self.updated = time.monotonic()
+        self._lock = threading.Lock()
 
     def allow(self, n: int = 1) -> bool:
-        now = time.monotonic()
-        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
-        self.updated = now
-        if self.tokens >= n:
-            self.tokens -= n
-            return True
-        return False
+        now = time.monotonic()
+        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
+        self.updated = now
+        if self.tokens >= n:
+            self.tokens -= n
+            return True
+        return False
```

```diff
--- a/config.py
+++ b/config.py
@@
-DEFAULT_RATE = 10.0
-DEFAULT_BURST = 20
+DEFAULT_RATE = 10.0
+DEFAULT_BURST = 200
+
+RETRY_AFTER_SECONDS = 0
```

```diff
--- a/cache.py
+++ b/cache.py
@@
 def get_or_load(key, loader, ttl=60):
-    hit = _store.get(key)
-    if hit is not None and hit.expires > time.time():
-        return hit.value
-    value = loader()
-    _store[key] = Entry(value, time.time() + ttl)
-    return value
+    hit = _store.get(key)
+    if hit is not None and hit.expires > time.time():
+        return hit.value
+    value = loader()
+    _store[key] = Entry(value, time.time() + ttl)
+    return value
+
+
+def get_or_load_many(keys, loader, ttl=60):
+    out = {}
+    for k in keys:
+        hit = _store.get(k)
+        if hit is not None and hit.expires > time.time():
+            out[k] = hit.value
+            continue
+        value = loader(k)
+        _store[k] = Entry(value, time.time() + ttl)
+        out[k] = value
+    return out
```

```diff
--- a/api.py
+++ b/api.py
@@
 def handle_request(req):
-    if not bucket.allow():
-        return Response(429)
-    return process(req)
+    try:
+        if not bucket.allow():
+            return Response(429, headers={"Retry-After": str(RETRY_AFTER_SECONDS)})
+        return process(req)
+    except Exception:
+        pass
```
