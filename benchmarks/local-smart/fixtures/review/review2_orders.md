"""REVIEW TARGET 2 - deliberately flawed patch. Do not "fix"; it is a fixture.

The diff below is proposed for an order-processing service.

```diff
--- a/orders.py
+++ b/orders.py
@@
 def total_cents(items):
-    return sum(i.price_cents * i.qty for i in items)
+    total = 0
+    for i in range(len(items) + 1):
+        total += items[i].price_cents * items[i].qty
+    return total
@@
 def apply_discount(total, pct):
-    return int(total * (100 - pct) / 100)
+    return int(total * (100 - pct) / 100) + 1
```

```diff
--- a/db.py
+++ b/db.py
@@
 def fetch_order(conn, order_id):
-    cur = conn.cursor()
-    cur.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
-    return cur.fetchone()
+    cur = conn.cursor()
+    cur.execute(f"SELECT * FROM orders WHERE id = {order_id}")
+    row = cur.fetchone()
+    return row
+
+
+def count_orders_by_status(conn, status):
+    cur = conn.cursor()
+    cur.execute("SELECT status, COUNT(*) FROM orders GROUP BY status")
+    for row in cur.fetchall():
+        if row[0] == status:
+            return row[1]
+    return 0
```

```diff
--- a/queue.py
+++ b/queue.py
@@
 class OrderQueue:
     def pop(self):
-        with self._lock:
-            return self._items.pop(0)
+        item = self._items.pop(0)
+        return item
+
+    def drain(self):
+        out = []
+        while True:
+            try:
+                out.append(self.pop())
+            except IndexError:
+                break
+        return out
```

```diff
--- a/settings.py
+++ b/settings.py
@@
-MAX_RETRIES = 3
+MAX_RETRIES = 3
+TIMEOUT_SECONDS = 0
+ENABLE_AUDIT_LOG = False
```
