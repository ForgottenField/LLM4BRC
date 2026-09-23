# LLM4BRC

**LLM-assisted false-positive detection and proof-of-concept generation for Clang Static Analyzer bug reports.**

> 本仓库把一份 Clang Static Analyzer（CSA）的 HTML 报告判定为 **TP / FP / UNKNOWN**，
> 并在判定为 TP 时生成可编译的 POC。判定过程是一个**有序的 10 阶段流水线**：前 8 阶段是确定性的
> 程序分析（PDG/SDG 反向切片、CFG 分支抽取、约束与冲突检查、路径空间压缩），
> 后 2 阶段用 LLM 做路径选择与模拟执行审计。
>
> 方法论的**唯一权威描述**在 [`docs/pipeline-stages.md`](docs/pipeline-stages.md)——本 README 只是索引与上手指南。

---

## 1. 它解决什么问题

Clang Static Analyzer 会报出「空指针解引用」「内存泄漏」「未初始化读」等 bug path，
但其中相当一部分是**误报**：报告给出的路径在真实调用方下不可达、前提自相矛盾、
或只是测试代码自己构造出来的输入。

本仓库不满足于「POC 能编译就说明是 TP」。它先**在枚举之前**用两条确定性闸门淘汰
自相矛盾的报告，再把剩余报告的**整条路径空间**收集起来、按 SDG 相关性压缩，
最后对每条可行路径生成 POC 并让 LLM **逐步对照 CSA 报告的路径做模拟执行**——
每一步都必须复现，否则该轮作废重生成。**一步对不上，`triggered` 也不接受为 TP。**

判定的三个出口：

| verdict | 含义 |
|---|---|
| **TP** | 报告路径真实可达，POC 能在模拟执行中触发报告所述的 bug |
| **FP** | 路径不可实现（前提自相矛盾 / 入口状态是 POC 自己制造的 / 有限域被逼入矛盾 / 全空间穷尽后无可行路径） |
| **UNKNOWN** | 路径空间过大只能有界采样，或模拟执行出现 unresolved——保守地不下结论 |

---

## 2. 十个阶段

```
report.html
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 阶段 1  报告解析与工程装配    HTML → ParsedReport + bug_path          │
│         校验 source_root / pdg_<project>.json / cfg_cache（fail-fast）│
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 2  路径分段              → SegmentInfo[]（按 (file, 行区间) 归函数）│
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 3  PDG 反向切片          → SliceMask（哪些节点与根因相关）        │
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 4  假设可行性            → hard_constraints + cumulative_state    │
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 5  分支过滤              → 每分支 consistent/contradictory/       │
│                                 insufficient（4 层冲突 + A/B/C 剪枝） │
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 6  确定性闸门（无 LLM，共用一个早返回出口）                       │
│         6a 报告自身路径把同一条件判成两个方向 ⇒ 不可实现               │
│         6b 入口状态制造：报告的 null 入口假设 + 自己的路径穿过函数体内  │
│            必然终止的致命检查 ⇒ 不可实现                               │
│                        │矛盾 ⇒ FP，直接返回，不生成 POC                │
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 7  语义有限域闸门（LLM）                                          │
│         问：有限域变量是否被逼入自相矛盾 ⇒ FP，直接返回                │
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 8  路径空间收集与压缩    → whole_path_bound / reduced_bound       │
│         （按 SDG 相关性分类 → 无关分支固定 → 互斥剪枝）                │
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 9  路径选择              冲突驱动重选 + 整路径组合枚举            │
│                                ↺ 每条可行路径交给阶段 10              │
├──────────────────────────────────────────────────────────────────────┤
│ 阶段 10 POC 验证与结论                                                 │
│         10a 源码上下文 → 10b 约束补全 → 10c 生成 POC（不真编译）       │
│         → 10d LLM 模拟执行 + 反馈循环（≤3 轮，逐步路径一致性审计）     │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
TP / FP / UNKNOWN  +  output/path_selection_<report>.json  +  poc_<stem>.cpp
```

阶段 6 与 7 是**三条「枚举前就返回 FP」的闸门**（6 的 6a/6b 是确定性的，7 用 LLM）。
阶段 9/10 内还有四种「推广」：路径空间压缩推广、结构性 FP 根因不变性推广、
leak 层一致性推广、路径探索聚合——它们让**一条路径上的结论**在满足 soundness 前置条件时
推广到整个相关空间（**单路径 FP 永不单独定案**）。

逐阶段的输入/输出/判定出口/代码位置/控制台输出样例，见
[`docs/pipeline-stages.md`](docs/pipeline-stages.md)。

---

## 3. 仓库结构

```
run_path_selection.py          # 端到端入口：10 阶段流水线 + CLI
src/llm_client/
  ├── base.py / factory.py     # LLMProvider 抽象层（deepseek / claude / gpt）
  ├── errors.py                # 统一异常层级（provider 不许抛裸 SDK 错误）
  ├── deepseek_provider.py     # 默认 provider（deepseek-chat）
  ├── fp_analysis/             # ★ 判定核心
  │   ├── pdg_*.py  slicer.py  slice_mask.py        # PDG/SDG + 切片
  │   ├── cfg_*.py  condition_extractor.py          # CFG 解析、分段、分支抽取
  │   ├── conflict_*.py  constraint_*.py            # 4 层冲突检查、约束挖掘/缓存
  │   ├── path_space.py  branch_relevance.py        # 路径空间收集与压缩
  │   ├── cause_invariance.py  structural_pruning.py
  │   ├── report_consistency.py                     # 阶段 6a
  │   ├── entry_state.py                            # 阶段 6b
  │   ├── domain_facts.py                           # 阶段 7 补充事实
  │   ├── simulation_verifier.py                    # 阶段 10d 模拟执行审计
  │   └── path_analyzer.py                          # LLM 驱动的选路/POC 生成
  └── csa_analysis/            # C++ 原生工具（PDG / 调用图 / 可达性 checker）
      ├── pdg/PDGBuilder.cpp      build.sh   ← pdg_<project>.json 的生产者
      ├── callgraph/              build.sh
      └── checker/                build.sh
tests/                         # pytest 套件（按子系统分组）
tools/                         # 依赖图构建 + 回归/复现脚本
docs/                          # 方法文档（见 §7）
```

数据产物（**gitignore，可重建**）：

| 产物 | 位置 | 生产者 | 消费者 |
|---|---|---|---|
| `pdg_<project>.json` | 仓库根目录 | `csa_analysis/pdg/`（`pdg_builder`） | 阶段 3 切片、阶段 5 分支抽取、函数归属 |
| `cfg_cache/<project>/*.json` | `cfg_cache/` | `cfg_feasibility` 离线 dump | 阶段 2/5 的 CFG 分支抽取（按需加载合并多 TU） |
| `constraint_cache/` | `constraint_cache/` | `constraint_miner` | 阶段 4/5 约束复用 |
| 运行产物 | `output/` | 流水线本身 | 人工阅读；可随时清空 |

---

## 4. 环境要求

- Linux
- **`python3.10`**（pytest 与 `PYTHONPATH` 均以 3.10 为准）
  > 注意：本机默认 `python3` 可能是 3.8，会在导入时抛
  > `TypeError: 'type' object is not subscriptable`（`llm_client/base.py` 里的 `list[dict[str, str]]`，
  > 该模块没有 `from __future__ import annotations`）。这个报错看起来完全不像 Python 版本问题——
  > 一律显式写 `python3.10`。
- `clang++-14` + `llvm-config-14`（构建 C++ 插件，动态链接 `libclang-cpp`）
- `cmake ≥ 3.16`
- DeepSeek API key（默认 provider）

包**没有安装**（无 `setup.py`/`pyproject.toml`）：测试需要 `PYTHONPATH=src`；
`run_path_selection.py` 自己把 `src/` 加进 `sys.path`，所以不需要。

```bash
# API key：放在仓库根的 .env（已 gitignore，绝不提交）
echo 'DEEPSEEK_API_KEY=sk-...' > .env
```

---

## 5. 快速开始

```bash
# 1) 测试套件（全绿基线）
PYTHONPATH=src python3.10 -m pytest tests/ -q

# 2) 单个测试文件 / 单个用例
PYTHONPATH=src python3.10 -m pytest tests/test_fp_analysis/test_conflict_checker.py
PYTHONPATH=src python3.10 -m pytest tests/test_factory.py::TestProviderFactory::test_create_gpt_provider

# 3) 端到端跑一份报告（会调 LLM）
python3.10 run_path_selection.py --report ~/csa_reports/project/faiss/reports/FP/report-index_read.cpp-read_index-484-1.html
python3.10 run_path_selection.py --source-root ~/csa_reports/project/faiss --report <report.html>

# 4) 批量：扫描 <projects-dir>/<project>/reports/{TP,FP}
python3.10 run_path_selection.py --project ~/csa_reports/project/faiss
python3.10 run_path_selection.py --scan-dir /path/to/reports --project faiss --max-reports 5
```

主要 CLI 参数：

| 参数 | 说明 |
|---|---|
| `--report <html>` | 单份报告 |
| `--project <name\|path>` | 不加 `--report` 时批量扫描该项目 `reports/{TP,FP}` |
| `--scan-dir <dir>` | 批量扫描任意目录下的 `report-*.html` |
| `--source-root <dir>` | 显式源根（覆盖 `--project` 推断） |
| `--projects-dir <dir>` | 语料基目录（默认 `~/csa_reports/project`） |
| `--pdg` / `--cfg-cache` | 显式指定产物 |
| `--max-path-attempts N` | 路径遍历阈值，超过则 UNKNOWN（默认 3） |
| `--path-space` | 打印每段可行路径空间的 ASCII 树 |
| `--output` | 单报告时是结果文件路径；批量时是**目录**（默认 `output/path_selection_<report>.json`） |
| `--batch-summary` | 批量聚合 JSON（默认 `output/batch_summary.json`），与 `--output` 分开以免互相覆盖 |

> **没有模式开关**：冲突驱动选路是唯一的选路机制，POC 验证恒开。
> 历史上的 `--verify` / `--no-verify` / `--v2` / `FP_USE_CONFLICT_LEARNING` 已全部删除，请不要再加。

---

## 6. 准备一个项目的「三件套」

`run_path_selection.py` **从不自动构建**产物：`pdg_<project>.json` 必须已存在
（缺失即报错退出，绝不替换成别的项目），CFG cache 只做查表（未命中打
`WARNING: no CFG cache for <file>`）。

按需构建（只建报告真正走到的 TU——种子来自报告本身，再加有界的被调闭包）：

```bash
# 按需：种子 = bug 宿主文件 + 报告路径上的每个 event 文件，闭包补跨 TU 被调者
python3.10 tools/build_project_deps.py --project protobuf \
    --on-demand ~/csa_reports/project/protobuf/reports

# 只看清单不构建（种子 TU / 被调闭包 / 编译参数 / 语法通过率）
python3.10 tools/build_project_deps.py --project protobuf --on-demand <dir> --dry-run

# 校验 PROJECT_SPECS 与 cfg_parser._CFG_PROJECT_CONFIGS 是否漂移，并列出每个 pdg_*.json 的 OK/STALE
python3.10 tools/build_project_deps.py --check-config
```

C++ 插件本身单独构建（产物落在各自的 `build/`）：

```bash
bash src/llm_client/csa_analysis/pdg/build.sh        # → build/pdg_builder → pdg_<project>.json
bash src/llm_client/csa_analysis/checker/build.sh    # 可达性 checker
bash src/llm_client/csa_analysis/callgraph/build.sh
```

⚠️ **产物版本必须与代码一致**：`pdg_<project>.json` 的 `Metadata["version"]` 需要等于
`pdg_models.PDG_ARTIFACT_VERSION`（当前 **1.3**）。陈旧产物**能正常解析**，只有那行声明会说它是旧的——
历史上 `pdg_protobuf.json` 停在 1.0 好几周，一整轮 21 份报告评估跑在缺了 42% 控制依赖边的图上。
`tools/build_project_deps.py` 拒绝在版本漂移时构建并删除不匹配的产物，`_resolve_paths` 拒绝分析它。

完整流程（compile db 的坑、子模块版本以 `git ls-tree` 为准、行号对齐必须「验证」而不是假设、
按需构建的三条不变量）见
[`docs/dependency-graph-build-guide.md`](docs/dependency-graph-build-guide.md)。

---

## 7. 文档

| 文档 | 内容 |
|---|---|
| [`docs/pipeline-stages.md`](docs/pipeline-stages.md) | **方法的唯一权威描述**：10 阶段逐条详解、三条枚举前 FP 出口、四种推广、verdict 总表、13 条设计不变量、回归工具与基线 |
| [`docs/dependency-graph-build-guide.md`](docs/dependency-graph-build-guide.md) | 为新项目构建 `pdg_<project>.json` + `cfg_cache/<project>/` 的统一方法（含按需构建策略） |
| [`docs/fp_verification_case_study.md`](docs/fp_verification_case_study.md) | 两个 FP 案例的全过程：「生成 POC → LLM 模拟执行 → 多轮反馈」如何把判定收敛到 FP |
| [`CLAUDE.md`](CLAUDE.md) | 面向在本仓库工作的 AI/开发者的架构索引与踩坑记录（含各条不变量背后的历史事故） |

---

## 8. 复现与回归

```bash
# 确定性阶段 1–6（无 LLM），打印每段 fn/file/CFG key/分支数
python3.10 tools/replay_steps1to6.py <reports...>

# 同一批报告跑两遍（A/B/C 剪枝 关 vs 开），对比节点/分支数与整路径 bound
python3.10 tools/abc_compare.py      <reports...>

# 阶段 6 的两条闸门（6b 入口状态制造 + 6a 矛盾检测）+ claim 直方图
python3.10 tools/probe_report_consistency.py <reports...>

# 按 ground truth 汇总 output/ 下的结果 JSON
python3.10 tools/summarize_faiss_eval.py output
```

**回归集**：`484-2`、`484-1`、`clone_index`（必须保持 2 段 / 4 分支 / bound 16）、
`fourcc`、`IndexFastScan`、`read_VectorTransform`、`test_merge`。

**阶段 6 基线**（任何改动都必须让它们保持不变）：

| 语料 | 6a 报告自相矛盾 | 6b 入口状态制造 |
|---|---|---|
| faiss 4 份 | `484-1` / `484-2` 各 1 处；`fourcc` / `clone_index` 各 0 | — |
| faiss 13 / aria2 11 / folly 8 份 | 0 | **全 0**（含 `clone_index`——那里的 null 来自真实调用方数据通路，属 TP） |
| protobuf 21 份 | 0 | **2**（`numbers.cc-SimpleAtob-{11,5}-1`） |

判定阶段 1–9 的计数基线：`IndexFastScan` 44 分支、`fourcc` 1152、`484-1` 34 段 10152 分支、
`484-2` 45 段 10136 分支、`read_VectorTransform` 119、`test_merge` 34、`clone_index` 2 段 4 分支 bound 16。

⚠️ 评估脚本按 `verdict_step` 里的**中文关键字**打标记（如 `"遍历超限" in step`）——
改阶段编号时必须保留这些关键字短语。

---

## 9. 语料

源码树与报告**不在本仓库内**，位于 `~/csa_reports/project/<name>/`：

```
~/csa_reports/project/<name>/
├── <项目源码树>            # 含 deps/ lib/ src/ .git CMakeLists.txt ... 之一
└── reports/
    ├── TP/report-*.html    # ground truth：真 bug
    └── FP/report-*.html    # ground truth：误报
```

测试通过 `CSA_PROJECTS_DIR`（默认 `~/csa_reports/project`）解析这些路径，**不存在即 skip 而不是失败**。
可用 `--projects-dir` 覆盖。

当前评估语料规模：faiss 13、aria2 11、folly 8、protobuf 21 份报告。

---

## 10. 关键设计不变量（摘要）

完整 13 条见 [`docs/pipeline-stages.md` §8](docs/pipeline-stages.md)。最容易踩的几条：

1. **无模式开关**——不要再加。
2. **单路径 FP 永不单独定案**——FP 只能来自「冲突式不可行」或「全空间穷尽/推广」。
3. **`insufficient` 是保守默认态**，不是「不可行」。
4. **产物存全量、只在渲染时截断**——绝不靠存一份被截断的产物（截断曾把五条不同的 `EXPECT_EQ` 压成同一段文本，凭空造出「同一条件」）。
5. **模拟审计**：`steps_checked` 必须等于渲染步数；解析降级 ⇒ 重生成、退化为 UNRESOLVED，**绝不翻 FP**。
6. **阶段 1 路径解析必须 fail-fast**——绝不把无法解析的项目静默替换成另一个。
7. **阶段 6b 的致命检查家族只收无条件编译的**（`assert`/`DCHECK`/`FAISS_ASSERT` 在 NDEBUG 下被编译掉，那正是 CSA 的 null 得以存活的原因）；调用点扫描**永远只是旁证**。

---

## 11. 面向开发者

```bash
# 全部测试
PYTHONPATH=src python3.10 -m pytest tests/ -q
```

在本仓库工作前请先读 [`CLAUDE.md`](CLAUDE.md)（架构索引 + 每条不变量背后的历史事故）
与 [`docs/pipeline-stages.md`](docs/pipeline-stages.md)（方法的权威定义）。
改动任何阶段前，先看 `docs/pipeline-stages.md` §9 的基线表——它们是防回归的判据。
