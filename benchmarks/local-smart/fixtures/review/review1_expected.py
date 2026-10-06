"""Ground truth for review1_limiter.md.

`matches` is a list of alternative keyword groups; an issue counts as DETECTED when
any alternative's keywords all appear in the model's reported issues (case-insensitive).
`severity` is the minimum acceptable severity for that issue.
"""

FIXTURE = "review1_limiter"

# Issues deliberately planted in the patch.
EXPECTED = [
    {
        "id": "lock_never_acquired",
        "desc": "A threading.Lock is created but allow() never acquires it, so the "
                "critical section is still racy.",
        "severity_at_least": "high",
        "matches": [
            ["lock", "never", "acquire"],
            ["lock", "not", "used"],
            ["lock", "unused"],
            ["没有", "加锁"],
            ["未", "加锁"],
        ],
    },
    {
        "id": "burst_raised_10x",
        "desc": "DEFAULT_BURST raised 20 -> 200, silently weakening the rate limit by 10x.",
        "severity_at_least": "medium",
        "matches": [
            ["burst", "200"],
            ["burst", "increase"],
            ["burst", "raised"],
            ["burst", "10x"],
            ["突发", "200"],
            ["限流", "放宽"],
        ],
    },
    {
        "id": "retry_after_zero",
        "desc": "RETRY_AFTER_SECONDS = 0 tells clients to retry immediately, causing a "
                "tight retry loop against a rate limiter.",
        "severity_at_least": "medium",
        "matches": [
            ["retry-after", "0"],
            ["retry_after", "0"],
            ["retry", "immediately"],
            ["重试", "0"],
            ["立即重试"],
        ],
    },
    {
        "id": "swallowed_exception",
        "desc": "except Exception: pass swallows all errors and returns None, hiding "
                "failures and returning a wrong response to the caller.",
        "severity_at_least": "high",
        "matches": [
            ["except", "pass"],
            ["swallow"],
            ["silent"],
            ["suppress"],
            ["吞", "异常"],
            ["静默"],
        ],
    },
    {
        "id": "n_plus_one_ttl",
        "desc": "get_or_load_many re-evaluates time.time() per key and re-implements "
                "get_or_load inline; duplicated logic that can drift, and it is not "
                "atomic against concurrent callers.",
        "severity_at_least": "low",
        "matches": [
            ["duplicat"],
            ["重复", "逻辑"],
            ["n+1", "ttl"],
            ["inline"],
            ["refactor"],
        ],
    },
]

# Things that are NOT problems. Reporting these is a false positive.
NOT_ISSUES = [
    {
        "id": "forgot_import_threading",
        "desc": "Claiming threading is not imported. The patch shows the change in "
                "context only; the import is not part of the diff.",
        "matches": [["not import", "threading"], ["missing import"]],
    },
    {
        "id": "lock_would_deadlock",
        "desc": "Claiming the new lock deadlocks. It is never acquired, so it cannot.",
        "matches": [["deadlock"]],
    },
    {
        "id": "ttl_parameter_unused",
        "desc": "Claiming ttl is unused in get_or_load. It is used.",
        "matches": [["ttl", "unused", "get_or_load"]],
    },
]
