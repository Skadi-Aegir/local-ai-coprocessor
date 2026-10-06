"""Ground truth for review2_orders.md."""

FIXTURE = "review2_orders"

EXPECTED = [
    {
        "id": "off_by_one_range",
        "desc": "range(len(items) + 1) iterates one past the end -> IndexError on "
                "every call with a non-empty list.",
        "severity_at_least": "critical",
        "matches": [
            ["len(items)", "+ 1"],
            ["len(items)", "+1"],
            ["off-by-one"],
            ["off by one"],
            ["indexerror"],
            ["越界"],
            ["多一"],
        ],
    },
    {
        "id": "sql_injection",
        "desc": "f-string interpolation of order_id into SQL -> SQL injection. The "
                "parameterised query was replaced with string formatting.",
        "severity_at_least": "critical",
        "matches": [
            ["sql", "inject"],
            ["f-string", "sql"],
            ["f\"select"],
            ["string", "format", "query"],
            ["拼接", "sql"],
            ["注入"],
        ],
    },
    {
        "id": "lock_removed",
        "desc": "The lock around _items.pop(0) was removed, making pop() racy and "
                "list.pop(0) unsafe under concurrent access.",
        "severity_at_least": "high",
        "matches": [
            ["lock", "remov"],
            ["no longer", "lock"],
            ["without", "lock"],
            ["race"],
            ["移除", "锁"],
            ["去掉", "锁"],
            ["并发", "不安全"],
        ],
    },
    {
        "id": "drain_swallows_indexerror",
        "desc": "drain() uses IndexError for control flow, so a genuine IndexError "
                "inside pop() would silently terminate the drain early.",
        "severity_at_least": "medium",
        "matches": [
            ["indexerror", "control flow"],
            ["swallow", "indexerror"],
            ["exception", "control flow"],
            ["异常", "控制流"],
            ["掩盖", "indexerror"],
        ],
    },
    {
        "id": "discount_plus_one",
        "desc": "+1 added to every discounted total, a silent one-cent overcharge on "
                "every order.",
        "severity_at_least": "high",
        "matches": [
            ["+ 1", "discount"],
            ["+1", "discount"],
            ["one cent"],
            ["off by one", "discount"],
            ["多收"],
            ["多加 1"],
            ["多加1"],
        ],
    },
    {
        "id": "timeout_zero",
        "desc": "TIMEOUT_SECONDS = 0 makes every request time out immediately (or "
                "disables the timeout entirely, depending on the client).",
        "severity_at_least": "high",
        "matches": [
            ["timeout", "0"],
            ["超时", "0"],
        ],
    },
    {
        "id": "audit_log_disabled",
        "desc": "ENABLE_AUDIT_LOG = False silently disables auditing, which is usually "
                "a compliance requirement for order processing.",
        "severity_at_least": "medium",
        "matches": [
            ["audit", "false"],
            ["audit", "disabl"],
            ["审计", "关闭"],
            ["审计", "禁用"],
        ],
    },
]

NOT_ISSUES = [
    {
        "id": "count_orders_slow",
        "desc": "Claiming count_orders_by_status fetches all rows so it is slow. It is "
                "inefficient but that is the only way to count by status given the "
                "query shown, and it is not a correctness bug.",
        "matches": [["all rows", "slow"], ["fetchall", "slow"]],
    },
    {
        "id": "max_retries_unchanged",
        "desc": "Reporting MAX_RETRIES=3 as a problem. It is unchanged from the original.",
        "matches": [["max_retries", "too low"], ["max_retries", "increase"]],
    },
]
