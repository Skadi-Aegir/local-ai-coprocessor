"""
tasks.py - fixed benchmark task set for the local-fast worker (Phase 2).

Design notes
------------
* Test B (JSON extraction) is generated from a seed list so that every repetition
  uses a DIFFERENT task_id / project / priority / worker / status. Running the same
  prompt 20x would measure caching, not schema adherence.
* Test C (routing) has a hand-labelled answer key. Several items are deliberately
  borderline (e.g. "summarise a log" is local-fast, "design a permission model" is
  not) so the score is informative rather than trivially 100%.
* Test E (code) includes executable checks: the generated Python is actually run,
  and the SQL is executed against an in-memory sqlite database.
"""

from __future__ import annotations

import random

ROUTES = [
    "local-fast",
    "local-smart",
    "deepseek",
    "claude",
    "codex",
    "local-ocr",
    "local-vision",
    "local-image",
    "human",
]

# --------------------------------------------------------------------------
# Test B - JSON extraction
# --------------------------------------------------------------------------

PROJECTS = ["eye-tracking", "agent-platform", "civ6-mod", "billing-api", "data-pipeline"]
WORKERS = ["local-fast", "local-smart", "deepseek", "claude", "codex"]
STATUSES = ["pending", "running", "blocked", "done"]


def make_extraction_cases(n: int = 20, seed: int = 20402) -> list[dict]:
    """Build n distinct natural-language task descriptions plus their gold JSON."""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        task_id = f"T-{200 + i * 7}"
        project = rng.choice(PROJECTS)
        priority = rng.choice([10, 25, 40, 55, 70, 80, 95])
        requires_gpu = rng.choice([True, False])
        worker = rng.choice(WORKERS)
        status = rng.choice(STATUSES)
        gpu_clause = "需要 GPU" if requires_gpu else "不需要 GPU"
        text = (
            f"任务 {task_id} 属于 {project} 项目，优先级 {priority}，"
            f"{gpu_clause}，预计由 {worker} 处理，目前状态 {status}。"
        )
        cases.append(
            {
                "text": text,
                "gold": {
                    "task_id": task_id,
                    "project": project,
                    "priority": priority,
                    "requires_gpu": requires_gpu,
                    "worker": worker,
                    "status": status,
                },
            }
        )
    return cases


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "project": {"type": "string"},
        "priority": {"type": "integer"},
        "requires_gpu": {"type": "boolean"},
        "worker": {"type": "string"},
        "status": {"type": "string"},
    },
    "required": ["task_id", "project", "priority", "requires_gpu", "worker", "status"],
    "additionalProperties": False,
}

EXTRACTION_SYSTEM = (
    "You extract structured data. Reply with ONLY a JSON object, no prose, no code fences. "
    'Schema: {"task_id": string, "project": string, "priority": integer, '
    '"requires_gpu": boolean, "worker": string, "status": string}'
)

# GBNF grammar enforcing the exact object shape (used to test the "grammar" fix)
EXTRACTION_GRAMMAR = r'''
root        ::= "{" ws "\"task_id\"" ws ":" ws string "," ws
                    "\"project\"" ws ":" ws string "," ws
                    "\"priority\"" ws ":" ws integer "," ws
                    "\"requires_gpu\"" ws ":" ws boolean "," ws
                    "\"worker\"" ws ":" ws string "," ws
                    "\"status\"" ws ":" ws string ws "}"
string      ::= "\"" [^"]* "\""
integer     ::= [0-9]+
boolean     ::= "true" | "false"
ws          ::= [ \t\n]*
'''.strip()


# --------------------------------------------------------------------------
# Test C - routing
# --------------------------------------------------------------------------

# (task description, expected route, why)
ROUTING_CASES: list[tuple[str, str, str]] = [
    ("把这段 3000 行的 Nginx access log 总结成 5 条要点", "local-fast", "日志摘要，确定性高"),
    ("把这个文件里的变量名 user_list 批量改成 users", "local-fast", "简单重命名"),
    ("识别这张扫描版合同截图里的文字", "local-ocr", "纯 OCR"),
    ("设计一个支持多租户 RBAC 与审计日志的权限系统", "claude", "安全关键 + 架构设计"),
    ("写一条 SQL 统计每个项目的任务数量和平均优先级", "local-fast", "简单 SQL"),
    ("这个角色权限改造涉及生产环境，如何保证不出安全事故", "claude", "生产安全关键"),
    ("根据这段描述生成一张赛博朋克风格的城市插画", "local-image", "图像生成"),
    ("分析 pytest 失败结果并判断是不是环境问题", "local-fast", "测试结果解释"),
    ("把这段 12000 token 的对话压缩成结构化 handoff", "local-fast", "上下文压缩"),
    ("重构整个后端服务，从单体拆成微服务", "codex", "大规模重构"),
    ("这个界面截图上为什么按钮是灰色的", "local-vision", "GUI 视觉理解"),
    ("把这段英文 README 翻译成中文", "local-fast", "简单翻译"),
    ("定位这个 Python traceback 的根因并给最小修复", "local-fast", "traceback 初判"),
    ("为一个 50 人团队设计三年的技术演进路线", "claude", "战略级架构问题"),
    ("解释这个复杂分布式事务的竞态条件并给出证明", "deepseek", "中高难度推理"),
    ("把 JSON 文件转换成 CSV", "local-fast", "简单代码"),
    ("这段代码有内存泄漏，需要深入分析 GC 行为", "deepseek", "需要深入分析"),
    ("读取这张图表并描述趋势", "local-vision", "图表理解"),
    ("这个决定会删除生产数据库，是否执行", "human", "危险操作需人工决策"),
    ("给这个新项目起一个包名并写一句话描述", "local-fast", "极简生成任务"),
    ("写一个跨平台的桌面端渲染引擎并保证 120fps", "codex", "高难度工程实现"),
    ("把会议录音转成文字纪要", "local-fast", "转录后处理（ASR 已由 local-asr 完成）"),
]

ROUTING_SYSTEM = (
    "You are a task router for a local AI coprocessor. "
    "Classify the task into exactly one route and reply with ONLY a JSON object:\n"
    '{"route": "<route>", "reason": "<short reason>", "confidence": <0.0-1.0>}\n'
    "Valid routes: " + ", ".join(ROUTES) + "\n"
    "Guidance: cheap deterministic work (summaries, extraction, simple code, translation, "
    "traceback triage) -> local-fast. Moderate reasoning that still needs a bigger model -> "
    "local-smart. Hard reasoning -> deepseek. Complex software engineering / large refactors -> "
    "codex or claude. Irreversible or dangerous decisions -> human. "
    "OCR / vision / image generation -> the matching local-* route."
)

ROUTING_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ROUTES},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["route", "reason", "confidence"],
    "additionalProperties": False,
}

ROUTING_GRAMMAR = r'''
root       ::= "{" ws "\"route\"" ws ":" ws route "," ws "\"reason\"" ws ":" ws string "," ws
                   "\"confidence\"" ws ":" ws number ws "}"
route      ::= "\"local-fast\"" | "\"local-smart\"" | "\"deepseek\"" | "\"claude\"" | "\"codex\""
             | "\"local-ocr\"" | "\"local-vision\"" | "\"local-image\"" | "\"human\""
string     ::= "\"" [^"]* "\""
number     ::= ("0" | "1") ("." [0-9]+)? | "0." [0-9]+
ws         ::= [ \t\n]*
'''.strip()


# --------------------------------------------------------------------------
# Test D - traceback triage
# --------------------------------------------------------------------------

TRACEBACK = '''Traceback (most recent call last):
  File "/srv/app/main.py", line 42, in <module>
    run_pipeline(config_path)
  File "/srv/app/pipeline.py", line 88, in run_pipeline
    cfg = load_config(config_path)
  File "/srv/app/config.py", line 27, in load_config
    return json.loads(Path(path).read_text())
  File "/usr/lib/python3.13/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
  File "/usr/lib/python3.13/json/decoder.py", line 345, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/usr/lib/python3.13/json/decoder.py", line 341, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)'''

TRACEBACK_EXPECT = {
    "exception": ["JSONDecodeError"],
    "origin_file": ["config.py"],
    "origin_line_any": ["27"],
    "keywords": ["empty", "空", "not valid json", "invalid json", "非 json", "不是 json", "读取", "read_text"],
}

TRACEBACK_SYSTEM = (
    "You triage Python tracebacks. Reply in this exact format:\n"
    "Exception: <type>\n"
    "Location: <file>:<line>\n"
    "Cause: <one sentence>\n"
    "Fix: <minimal fix, with code if short>"
)


# --------------------------------------------------------------------------
# Test E - simple code
# --------------------------------------------------------------------------

CODE_CASES = [
    {
        "name": "python_json_to_csv",
        "system": "You write short, correct Python. Reply with a single fenced code block, nothing else.",
        "prompt": (
            "Write a Python function `to_csv(records, path)` that takes a list of dicts "
            "with identical keys and writes them to a CSV file at `path` with a header row. "
            "Use only the standard library. The function must handle an empty list by "
            "writing just the header (which will therefore be unknown - in that case write "
            "an empty file)."
        ),
        "extract": "python",
        "kind": "python",
        "test": (
            "import csv, importlib.util, io, os, sys\n"
            "src = open(sys.argv[1], encoding='utf-8').read()\n"
            "ns = {}\n"
            "exec(src, ns)\n"
            "fn = None\n"
            "for k, v in ns.items():\n"
            "    if callable(v) and k == 'to_csv':\n"
            "        fn = v\n"
            "assert fn is not None, 'to_csv not defined'\n"
            "import tempfile, pathlib\n"
            "d = pathlib.Path(tempfile.mkdtemp())\n"
            "p = d / 'out.csv'\n"
            "fn([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], str(p))\n"
            "rows = list(csv.reader(open(p, newline='', encoding='utf-8')))\n"
            "assert rows[0] == ['a', 'b'], f'header wrong: {rows[0]}'\n"
            "assert rows[1] == ['1', '2'], f'row1 wrong: {rows[1]}'\n"
            "assert len(rows) == 3, f'row count wrong: {len(rows)}'\n"
            "p2 = d / 'empty.csv'\n"
            "fn([], str(p2))\n"
            "print('PASS')\n"
        ),
    },
    {
        "name": "shell_batch_rename",
        "system": "You write short, correct shell scripts. Reply with a single fenced code block, nothing else.",
        "prompt": (
            "Write a bash script that renames every file matching `*.jpeg` in the current "
            "directory to have a `.jpg` extension. It must be safe: skip if the target name "
            "already exists, and print what it does. Use only standard tools."
        ),
        "extract": "bash",
        "kind": "shell",
        "test": None,  # checked structurally
    },
    {
        "name": "sql_group_by",
        "system": "You write short, correct SQL. Reply with a single fenced code block, nothing else.",
        "prompt": (
            "Given a table `tasks(id INTEGER, project TEXT, priority INTEGER, status TEXT)`, "
            "write one SQL query returning, for every project, the number of tasks and the "
            "average priority, ordered by task count descending, only including projects "
            "that have more than 1 task. Name the columns `project`, `task_count`, `avg_priority`."
        ),
        "extract": "sql",
        "kind": "sql",
        "test": (
            "import sys, sqlite3, re\n"
            "src = open(sys.argv[1], encoding='utf-8').read()\n"
            "con = sqlite3.connect(':memory:')\n"
            "con.execute('CREATE TABLE tasks(id INTEGER, project TEXT, priority INTEGER, status TEXT)')\n"
            # NOTE: exactly one project ('b') has more than one task. If two did, a
            # correct HAVING COUNT(*) > 1 would return two rows and would contradict
            # the single-row assertion below. Keep this shape when editing.
            "con.executemany('INSERT INTO tasks VALUES (?,?,?,?)', [\n"
            "  (1,'a',10,'done'),(2,'b',30,'done'),(3,'b',40,'done'),\n"
            "  (4,'b',50,'done'),(5,'c',60,'done')])\n"
            "cur = con.execute(src)\n"
            "cols = [d[0].lower() for d in cur.description]\n"
            "rows = cur.fetchall()\n"
            "assert cols == ['project','task_count','avg_priority'], f'columns wrong: {cols}'\n"
            "assert len(rows) == 1, f'should exclude single-task projects: {rows}'\n"
            "assert rows[0][0] == 'b', f'order wrong: {rows}'\n"
            "assert rows[0][1] == 3, f'count wrong: {rows}'\n"
            "assert abs(rows[0][2] - 40.0) < 1e-6, f'avg wrong: {rows}'\n"
            "print('PASS')\n"
        ),
    },
    {
        "name": "python_pytest",
        "system": "You write short, correct Python tests. Reply with a single fenced code block, nothing else.",
        "prompt": (
            "Write a pytest test module for a function `parse_duration(s)` that converts "
            "strings like '1h30m', '45s', '2m' into a number of seconds. Cover: a plain "
            "seconds value, a minutes+seconds value, an hours+minutes value, and that an "
            "invalid string raises ValueError. Use `pytest.raises`."
        ),
        "extract": "python",
        "kind": "python",
        "test": None,  # structural check only (semantics depend on an unimplemented function)
    },
]


# --------------------------------------------------------------------------
# Test F - context compression (handoff)
# --------------------------------------------------------------------------

HANDOFF_SECTIONS = [
    "Goal",
    "Current state",
    "Completed",
    "Failed",
    "Important files",
    "Important decisions",
    "Tests",
    "Open questions",
    "Next step",
]

HANDOFF_SYSTEM = (
    "You compress long engineering context into a structured handoff another agent can "
    "resume from. Reply with exactly these sections, each as a markdown heading, in order:\n"
    + "\n".join(f"## {s}" for s in HANDOFF_SECTIONS)
    + "\nBe concrete: include file paths, decision reasons and blockers. Do not invent facts."
)
