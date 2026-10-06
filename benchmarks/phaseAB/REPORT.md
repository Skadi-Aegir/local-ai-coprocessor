# Phase A/B 收尾验收 — 施工报告

**施工方：DSH ｜ 验收方：Astra**
**仓库：** `D:\MultiAgent`（master）
**基线：** 审查修复分支 `claude/workspace-content-review-be429a`（`a83efe9`）→ 合并提交 `4166fdd`
**本轮提交：** `8b5f00d`（步骤 2–3）、`568799d`（步骤 4）、`ee2e0be`（步骤 5–6）
**日期：** 2026-10-05

**本报告不给 Phase A 或 B 的 PASS/FAIL 总评。** 硬性条件逐条列出是否满足与证据路径，判定权在 Astra。

---

## 0. 硬性条件逐条核对

| # | 硬性条件 | 是否满足 | 证据 |
|---|---|---|---|
| A-1 | 当前资料可**可重复重建** | **是** | 第二次独立构建 `same_index=true`、`vectors_bit_identical=true`、`max_abs_diff=0.0`、`min_row_cosine=0.99999976` → `AB-03-build/a_index_build2.json` |
| A-2 | 可**回退** | **是** | `vector/qa/current.prev` = `259fae05785b433c`；回退路径 `--switch-only --out vector/qa/current.prev` 已有测试覆盖 |
| A-3 | 引用对应**被索引的实际文件版本** | **是** | run1 中每一条引用均 `verified`（`hash_scheme sha256-lf-v1`）；`review.md` 另手工复核 h03/h07/h09 的引用行 |
| A-4 | 逐题结果**可审计** | **是** | `heldout_v1/results/run1.json` + `run1_audit.json`（审计 exit 0）+ `review.md` |
| A-5 | **无无依据陈述** | **是** | 人工复核 12 题：0 条无支持陈述 → `review.md` |
| A-6 | **无无效引用** | **是** | 全部 `verified`，0 条越界 |
| A-7 | **无 OOM** | **是** | 全程无 OOM、无服务崩溃 |
| A-8 | 未观察到 **2048 MiB 余量突破** | **是** | A 运行最低空闲 **5208 MiB**；B 运行最低 **5175 MiB**；构建期间最低 **4012–4017 MiB** |
| 边-1 | 余量 2048 未降低 | **是** | `vector/qa/current/manifest.json` 与报告均记 `margin_mb: 2048` |
| 边-2 | reranker 瞬时 1800 / 默认预算阈值未改 | **是** | 本轮未改任何默认值；`safety_margin_mb: int = 2048` 原样 |
| 边-3 | 未新增/下载/替换模型 | **是** | 无下载动作 |
| 边-4 | 未改驱动/CUDA/页面文件/安全/Qwen-Image | **是** | 无此类动作 |
| 边-5 | 仅绑定 127.0.0.1 | **是** | `ask_server.py` 拒绝非回环 host；8111/8113/8115 均 127.0.0.1 |
| 边-6 | 冻结题集一字未改 | **是** | `heldout_v1` `sha256-lf-v1 = fa1cf6d5…2084` 运行前校验通过；`p56` 未改 |
| 边-7 | 未覆盖历史结果 | **是** | 新产出写入 `AB-*` 新 run-id 目录；`AB-02-deployed.json`（无效首跑）保留 |
| 边-8 | 子进程输出一律写文件、无管道 | **是** | 全部 `*>` / `-RedirectStandardOutput`；未用 `capture_output=True` |
| 边-9 | 进程归属按 `Win32_Process.CommandLine` 判定 | **是** | 步骤 0 与收尾均用 CommandLine 核对 |
| 边-10 | QA 路径 reranker 非常驻 | **是** | 8114 全程 down |
| 边-11 | 未提交 `agentvoice/` | **是** | 全部提交按路径显式 `git add`；收尾 `?? agentvoice/` 仍是未跟踪 |
| 边-12 | 未进入 C / Phase 6 | **是** | 未实施 |
| 边-13 | 未放宽任何门槛 | **是** | 无条件被调整；发现的截断缺陷**未修**，正因不得事后改动 |

**未发生任何"立即停止"条件**：无余量突破、无 OOM、无崩溃、P1-11 为 `released`、构建
`restart_required=false`、无离线测试失败、无构建校验失败、无泄漏、无冻结哈希不符、审计未拒绝、
无合并冲突。

---

## 1. 步骤 0 — 环境快照（只读）

`AB-00-env/env_before.json`

| 项 | 值 |
|---|---|
| GPU | 12227 total / 2386 used / **9559 free** MiB |
| 8111 / 8113 / 8114 / 8115 | 快照时**全部未监听** |
| HEAD | `35cf3b1`（master），分支头 `a83efe9` **与规定基线一致** |
| D: 可用 | **131.1 GB** / 652.9 GB |
| 系统可用提交内存 | 40545 MB |
| `config/local-fast.yaml` | `runtime.context=8192`、`parallel=1`，**与预期一致**（QA token 预算依赖此二值） |
| 未跟踪 | `agentvoice/`（未读写未提交）、`benchmarks/phaseAB/`（本轮证据） |

## 2. 步骤 1 — 合并与离线测试

`git merge --no-ff` **无冲突**。合并后五个离线测试全部通过，逐项与预期一致：
`test_qa_path.py` 68、`test_a_build_index.py` 59、`a_audit_heldout.py --selftest` 通过、
`test_chunker.py` 38、`test_index_store.py` 42。

> 一处**我方失误**：首次以 PowerShell 数组传 `--selftest` 被逐字符展开成
> `- - s e l f t e s t`，argparse 报错退出 2。这是调用方式错误，**不是测试失败**；改为直接传参后
> exit 0、`audit self-test passed`。

## 3. 步骤 2 — P1-11 在**已部署**的 8113 上验证

8113 从本树重启。**`/stats` 出现 `memory` 字段**，这即"运行修复后代码"的判据：
`torch_reserved_mb 1160.0`、`release_threshold_mb 256`。

`p1_11_service_smoke.py --port 8113 --run-id AB-02-deployed2`：

| 字段 | 值 |
|---|---|
| `verdict` | **`released`** |
| `retained_mb` | **0.0** |
| `cache_releases_during_embed` | **1** |
| `peak_above_weights_mb` | 1893.6（随后释放） |
| docs / 最长 chunk | 8 / 6456 chars |
| 耗时 | 1.05 s |

余量全程守住。**P1-11 因此在部署端而不只是在隔离实验中成立。**

> `AB-02-deployed.json`（首跑）**保留且标记为无效**：当时一次 `job_kill` 把 8113 一并停掉，导致
> connection-refused。有效跑是 `AB-02-deployed2.json`。此失败**如实保留**，未删除。

## 4. 步骤 3 — 重建 QA 索引（关闭 P1-13）

前置条件均满足（8113 跑修复代码、8114 非常驻、无进程占用 `vector/qa/current`）。构建退出码 **0**：

| 项 | 值 |
|---|---|
| `n_files` / `n_chunks` / excluded | **48 / 536** / 33 — **与预期完全一致** |
| `validation_passed` | **true** |
| `n_answer_key_leaks` | **0** |
| `question_sets_checked` | **p56: 10 且 heldout_v1: 12**（两者同时包含） |
| `build_vram.margin_held` | **true**，`free_min_mb` **4017**（构建前 4967） |
| `retention.restart_required` | **false**，`retained_mb` **0.0** |
| `hash_scheme` / `index_version_scheme` | **`sha256-lf-v1`** / **2** |
| `chunks_sha256` / `vectors_sha256` | `579e5b43a9f62cbf` / `9bcf0dbad9508cf6`（**均存在**） |
| `corpus_files_not_at_commit` | **[] 空** |
| `working_tree_dirty` | true — 来自 `agentvoice/` 与本轮未跟踪证据，**非语料** |
| `index_version` | **`488f54d755224c5b`** |

**未出现** "previous index is kept at current.old-…" 提示（无此类目录产生）。
`current.prev` 现为 **`259fae05785b433c`（干净版本）**，被污染的 `a3e3c050636f838e` **已不再占任何槽位**
— **P1-13 由本次切换关闭**。

**可复现性（AD-17）**：`--compare-to` 按内容比较，`index_version_equal=true`、
`chunks_equal=true`、`vectors_bit_identical=**true**`、`max_abs_diff=**0.0**`、
`min_row_cosine=0.99999976`、`same_index=**true**`。未调整批大小。
派生的 `vector/qa/staging2` 已删除（比较结果已入报告）。

**CLI 冒烟**（非 heldout 题）：引用 `verified`，索引显示 `488f54d755224c5b`，5.372 s。

## 5. 步骤 4 — Phase A 验收运行

运行条件与 `questions.md` 预登记一致：新索引、prompt `qa-v2`、8113 跑修复代码、8111 up、8114 非常驻。
**只跑一次**，exit 0，38 s。

**最小空闲显存 5208 MiB**（154 样本 @200 ms，`mem_total − mem_used`）→ `AB-04-heldout/vram.csv`。

### 审计（exit 0）

9 题 answered（全部 `answered_needs_support_review`）；h10–h12 全部 `correct_refusal`。
9 道可答题的**期望证据文件全部 RETRIEVED 与 SHOWN**；关键事实 `h01 2/2, h02 2/2, h03 4/4, h04 2/2,
h05 6/6, h06 5/5, h07 3/3, h08 3/3, h09 3/3`。

**一处差异如实记录**：h03 的 expected-evidence 列为 `-` —— 答案引用的是 reranker 配置与
`docs/reranking-architecture.md`，而非 `expected.json` 指定的文件。其陈述**由所引 chunk 支持**，
故仍记 complete，但差异**记录在案、未抹平**。

### 逐题复核（`heldout_v1/review.md`）

| 类别 | 数 | 题号 |
|---|---:|---|
| 完整 | **9** | h01–h09 |
| 部分 | **0** | — |
| 错误 | **0** | — |
| 正确拒答 | **3** | h10 h11 h12 |
| **错误拒答** | **0** | — |
| 出错 | **0** | — |
| 无支持陈述 | **0** | — |
| 无效引用 | **0** | — |

**h01–h09 的完整 9 题、部分 0 题、错误拒答 0 题。** 不计算通过百分比。

**被取代旧值**：**没有任何回答把被取代的旧值当作当前值。** h09 同时给出 2000 与 8000 并标明各自身份
（8000 为当前、2000 已弃用）；h07 引用的是 P55R-01 之后的策略（被引文件自带 2026-10-05 更正注记）；
h08 引用更正后的分阶段数字。

**h01–h09 错误拒答原因**：无（0 题），故无需分类。

### 发现的缺陷（**报告，未修**）

`cli/ask.py` 发送 `max_tokens = MAX_ANSWER_TOKENS = 256`（L82、L564），且**不回读
`finish_reason`**，因此被长度截断的回答在记录中与正常结束无法区分。

实测：**h05 恰好用满 256 completion tokens，行文在句中截断**（可见文本以
`"The rollback mechanism involves reverting the routing change"` 结束，无终止符）；
P56 回归的 `p05`、`p07` 同样是 256 tokens、同样断在词中。其余 8 道 heldout 与 5 道回归在终止符处结束。

**h05 仍记 complete 而非 partial**，因为 6 条期望事实**全部出现且各自有引用** —— 截断切掉的是收尾行文，
不是证据。之所以仍报告：一个**静默截断**的 256-token 上限是对该工具的真实限制，而"关键事实齐全"
会把这层掩盖掉。

**未作任何修改**：本阶段禁止在看到本题集结果后改动代码、prompt、语料、阈值或评分规则，
故 `MAX_ANSWER_TOKENS` **保持 256**，交由 Astra 决定，修复应属于**不读取本题集结果**的轮次。

### P56 回归（同条件；行为回归，非质量对比）

`AB-04-p56/p56_answers.json`：**answered 7 / refused 3 / errors 0**（此前 6 / 4 / 0）。
**一处判定变化：q05 `REFUSED → ANSWERED`。**

q05 正是此前"top-5 未取齐证据（且语料缺端口 8114）"那题。现以 2 条 `verified` 引用作答，覆盖
别名、`config/local-fast.yaml`、host `127.0.0.1`、port `8111` —— **trace 明显更完整**，但作为完整
trace **仍不完整：未覆盖 8113 嵌入服务**，且它是被 256 token 截断的三条之一。

**如实说明、不辩解**：q05 此前的分类来自**不同的索引**与**2000 字符打包上限**，两者均已改变，
故本次变化不能直接读作"检索能力提升"。引用条数亦有变化（q01 1→3、q02 3→2、q03 1→2、q04 3→2、
q06 3→2、q07 3→2），原因是打包与引用规则改变。索引、prompt、打包三者皆变，
**这只是一次行为回归检查**。全部引用 `verified`；延迟 0.926–5.927 s。

## 6. 步骤 5 — Phase B 复验

前置条件同步骤 4，另以新索引启动 `cli/ask_server.py`（127.0.0.1:8115，输出写文件）。

### `GET /health` 核对

| 检查项 | 期望 | 实际 | 满足 |
|---|---|---|---|
| `index_version` | 新版本 | `488f54d755224c5b` | **是** |
| `index_chunks` 三数一致 | 一致 | `manifest 536 / vectors 536 / metadata 536` | **是** |
| `index_hash_scheme` | `sha256-lf-v1` | `sha256-lf-v1` | **是** |
| `limits.max_executing` | 1 | **1** | **是** |

另记录：`max_concurrency 2`、`max_question_chars 1000`、`max_body_bytes 65536`、
上游超时 `embed 30 / tokenize 15 / generate 90`、`prompt_version qa-v2`、`reranker disabled`。

### 客户端场景（`AB-05-b/b_e2e.json`，exit 0）

| 场景 | 状态码 / 结果 | 耗时 | 与预期的差异 |
|---|---|---|---|
| 正常回答 | 200 answered，3 条引用全 `verified` | 13.823 s | 无 |
| 语料无答案 | 200 refused，0 引用 | 0.324 s | 无 |
| 索引覆盖 | **400 `index_override_rejected`** | — | 无 |
| **过期索引** | 200 refused；被改文件报 `stale`，另两条 `verified` | — | 无 |
| **服务不可用** | **503 `upstream_unavailable`** | 2.043 s | **与预期一致**（旧报告为 500） |
| 恢复 | **10.1 s**，`health=ok`，探针题 answered | — | 无 |
| 问题过长 | 413 `question_too_long` | — | 无 |
| **并发** | `[429,429,429,429,200,200]`，n_429=4 | — | **与预期一致**；两个 200 依次执行完 |

**过期索引场景确实被验证到**：被改动的 `docs/models/local-fast.md:132-148` **出现在该次回答的引用中**
并报 `stale`，故无需对"实际被引用的那个文件"补测。

**文件按字节恢复**：`git diff --exit-code docs/models/local-fast.md` → **exit 0**。

**余量**：最低空闲 **5175 MiB**（156 样本），守住。无 OOM。
**结束 inflight**：`current = 0`，`peak = 2`。

原预期两处差异**均已成立**：503 取代 500；6 并发得 2×200 + 4×429 且两个 200 依次完成。

### 补充三项（`AB-05-b/supplementary.json`）

| 用例 | 期望 | 实际 | 满足 |
|---|---|---|---|
| 只发头 + `Content-Length: 70000`，**不发体** | 413 `body_too_large` | **413 `body_too_large`**，0.022 s | **是** |
| 体为 `[1,2]` | 400 `bad_request` | **400 `bad_request`** | **是** |
| 体含 `index` 键 | 400 `index_override_rejected` | **400 `index_override_rejected`** | **是** |

第一项**在不发送请求体的情况下**即被拒绝，证明大小检查发生在读取体之前（否则会阻塞等待 70000 字节）。

> 一处**我方失误**：首版原始套接字读取器在读到头部终止符即关闭，未读响应体，导致 `kind=None`
> 而误判失败；首跑结果保留为 `supplementary.first-attempt.json`。按 `Content-Length` 读全后三项全通过。

**结束后已停止 8115**（按需启动，不常驻）。

## 7. 步骤 6 — Phase Gate

重建之后**再跑一遍全部离线测试**：五项全部 exit 0，与步骤 1 一致（68 / 59 / selftest / 38 / 42）。

## 8. 各步骤最小空闲显存

| 步骤 | 最小空闲 | 样本 | 余量守住 |
|---|---:|---:|---|
| 构建（步骤 3） | **4017 MiB** | 53 | 是 |
| 复现构建（步骤 3b） | **4012 MiB** | — | 是 |
| heldout_v1 run1（步骤 4） | **5208 MiB** | 154 @200 ms | 是 |
| Phase B 复验（步骤 5） | **5175 MiB** | 156 @200 ms | 是 |

## 9. 偏离计划、失败与回滚

| 项 | 说明 |
|---|---|
| 无合并冲突 | 未触发回滚 |
| 无索引回滚 | 新构建一次通过校验并切换 |
| **无代码回滚** | — |
| 失败 1 | `a_audit_heldout.py --selftest` 首跑 exit 2 —— **我方 PowerShell 传参错误**，非测试失败；改正后通过 |
| 失败 2 | `AB-02-deployed.json` connection-refused —— `job_kill` 连带停掉 8113；**保留为无效首跑**，重跑为 `AB-02-deployed2.json` |
| 失败 3 | 补充检查首跑误判 —— 原始套接字未读响应体；**保留首跑**，修正读取后三项全通过 |
| 未完成项 | **无**（步骤 0–6 全部执行） |
| 未修复项 | `MAX_ANSWER_TOKENS=256` 静默截断 —— **故意未修**，见 §5 |

**任何"立即停止"条件均未发生。**

## 10. 下一阶段所需的只读事实（未施工、未下载）

| 项 | 值 |
|---|---|
| **D: 盘可用空间** | **131.1 GB**（共 652.9 GB） |
| **8111 与 8113 同时常驻时的实测空闲显存** | **4913 MiB**（`nvidia-smi`，8114/8115 均 down） |
| **P1-6** | **仍开放，等待 Astra**：三模型常驻时 `152 MiB usable < 600 MiB required`，10/10 拒绝；替代方案（fast+embedding 常驻、3431 MiB usable）已提交为 AD-12，**未采纳** |
| **P2-7** | **仍开放**：0.30 证据下限**从未触发**（ERRATA E23）；语料无答案时的拒答完全依赖 4B 模型遵循 prompt |
| **P2-8** | **仍开放**：引用门只查身份与范围，**不查支持**；裸 `SOURCE n` 连 "SOURCE 2 does not say…" 也计入 |
| **P3-6** | **仍开放**：reranker 仅在 `close()` 释放，同 P1-11 的旧行为；本轮**未改**，因其预算与 P55R 共驻数字都基于该行为测得 |

## 11. 边界与未做事项

未进入 **C（跨文档多跳）**、未进入 **Phase 6 OCR**、未接入 **reranker**、未改 **P2-7/P2-8 证据阈值与引用门**、
未做**跨进程账本（P1-10）**、未建**统一 Gateway**。未新增/下载模型，未改驱动/CUDA/页面文件/安全/Qwen-Image，
未开放公网，未覆盖历史结果，`agentvoice/` 未读写未提交。
