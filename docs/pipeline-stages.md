# CSA 报告分类流水线的十个阶段

本仓库把一份 Clang Static Analyzer (CSA) 的 HTML 报告判定为 **TP / FP / UNKNOWN**，并在 TP 时生成
可编译的 POC。本文是该方法的**唯一权威描述**：十个阶段的顺序、每阶段的输入输出、判定出口、
以及代码位置。`CLAUDE.md` 只保留索引，细节以本文为准。

> 行号锚点均针对基线提交 `6017755`；重构后请以**函数名**为准（行号会漂移）。

---

## 0. 总览

```
                 ┌──────────────────────────────────────────────────────┐
  report.html ──▶│ 阶段 1  报告解析与工程装配                            │
                 │    HTML → ParsedReport + bug_path                     │
                 │    校验 source_root / pdg_<proj>.json / cfg_cache     │
                 └───────────────────────┬──────────────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ 阶段 2  路径分段  →  SegmentInfo[]                    │
                 └───────────────────────┬──────────────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ 阶段 3  PDG 反向切片  →  SliceMask                    │
                 └───────────────────────┬──────────────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ 阶段 4  假设可行性  →  hard_constraints + state       │
                 └───────────────────────┬──────────────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ 阶段 5  分支过滤  →  consistent/contradictory/        │
                 │                     insufficient（4 层冲突 + A/B/C）   │
                 └───────────────────────┬──────────────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────────────┐
        │ 阶段 6  确定性闸门（无 LLM），共用一个早返回出口               │
        │   6a 报告自身路径把同一条件判成两个方向 ⇒ 不可实现             │
        │   6b 入口状态制造：报告的 null 入口假设 + 自己的路径穿过       │
        │      函数体内必然终止的致命检查 ⇒ 不可实现                     │
        └───────────────┬────────────────────────────┬───────────────────┘
                  矛盾 ✗ │                            │ 均不成立
                        ▼                            ▼
                   ┌─────────┐      ┌────────────────────────────────────┐
                   │  FP     │      │ 阶段 7  语义有限域闸门（LLM）      │
                   └─────────┘      │   （前置：domain supplement 事实） │
                                    └───────┬──────────────┬─────────────┘
                                      矛盾 ✗ │              │ 无矛盾
                                            ▼              ▼
                                       ┌─────────┐  ┌───────────────────────────┐
                                       │  FP     │  │ 阶段 8  路径空间收集与压缩│
                                       └─────────┘  │  → bound / reduced_bound  │
                                                    └───────────┬───────────────┘
                                                                ▼
                                    ┌────────────────────────────────────────┐
                                    │ 阶段 9  路径选择（组合枚举 + 重选）    │
                                    │   ↺ 每条可行路径交给阶段 10            │
                                    └───────────┬────────────────────────────┘
                                                ▼
                                    ┌────────────────────────────────────────┐
                                    │ 阶段 10 POC 验证与结论                 │
                                    │  约束补全→生成 POC→审计→模拟执行+反馈  │
                                    └───────────┬────────────────────────────┘
                                                ▼
                                   TP  /  FP  /  UNKNOWN  +  result JSON
```

---

## 1. 阶段一览

| # | 阶段 | 输入 | 产出 | 代码位置 |
|---|------|------|------|----------|
| 1 | 报告解析与工程装配 | `report.html` + `--project`/`--source-root` | `ParsedReport`, `bug_path`, `SDG`, `CFGCacheSet` | `main`, `_resolve_paths`, `_looks_like_project_root`, `parse_report`, `load_sdg`, `_resolve_cfg_cache` |
| 2 | 路径分段 | `ParsedReport`, `SDG` | `SegmentInfo[]`（每段带 `function_file`/`function_name`） | `segment_path`（`cfg_feasibility`）, `CFGCacheSet.ensure` |
| 3 | PDG 反向切片 | `SDG`, `bug_path` | `SliceMask`, `SliceResult` | `build_slice_mask`, `path_seed_extractor`, `slicer`, `slice_mask` |
| 4 | 假设可行性 | `ParsedReport`, `SDG` | `hard_constraints`, `cumulative_state`（+ `FPAnalyzer`） | `FPAnalyzer.check_assumptions_feasibility` |
| 5 | 分支过滤 | 段 + CFG cache + `SliceMask` | 每段 `consistent`/`contradictory`/`insufficient` 分支 | `run_branch_filter`, `CFGBranchExtractor`, `ConflictChecker` |
| 6 | **确定性一致性闸门**（6a 自相矛盾 / 6b 入口状态制造） | 段的分支树 + 报告事件 + bug 文件 | **FP，直接返回** | `report_consistency.detect_report_contradictions`, `entry_state.detect_fabricated_entry_state` |
| 7 | **语义有限域闸门** | `ParsedReport` + domain facts | **FP，直接返回** | `extract_domain_facts`/`format_domain_facts`, `FPAnalyzer.semantic_fp_reason` |
| 8 | 路径空间收集与压缩 | 段的可行候选 | `whole_path_bound`, `node_relevance`, CI 分组 | `collect_path_space`, `classify_branch_relevance`, `reduce_path_space`, `annotate_ci_groups`, `iter_combinations` |
| 9 | 路径选择 | 压缩后的路径空间 | 可行 `selections`（或各推广结论） | `run_path_selection`, `FPAnalyzer._select_feasible_path_segments`, `_reselect_conflicting` |
| 10 | POC 验证与结论 | `selections` | **TP / FP / UNKNOWN** + `poc_*.cpp` | `_verify_path`, `FPAnalyzer._complete_constraints`/`generate_poc`, `SimulationVerifier` |

阶段 1–3 是**纯确定性装配**；4 有 LLM；5 是确定性判定 + LLM 复核；
**6（6a/6b）与 7 是三条"枚举前就返回 FP"的闸门**（6 的两条是确定性的，见 §3）；
8 纯确定性；9–10 是本方法的判定核心。

阶段 6 的两条判据共享一个早返回出口：命中任一条 ⇒ 结果 JSON 带各自的证据块
（`report_self_contradiction` / `entry_state_fabrication`）、**无 `poc_result`**、在阶段 8 之前返回。
两条都不命中才继续。

---

## 2. 逐阶段详解

### 阶段 1 — 报告解析与工程装配

**做什么**：解析 CSA HTML → `ParsedReport`（bug 类型、入口函数、bug 文件、`path_events`），
并向上装配三份数据：

- `pdg_<project>.json`（SDG，由 C++ PDG builder 产出）
- `cfg_cache/<project>/*.json`（按 `bug_file` 解析出入口函数对应的那份）
- `source_root`（源码根）

**关键约束（防静默错跑）**：源根与 PDG 都**必须显式可解析**。
`_resolve_paths` 只在推断出的根目录带有
`deps/ lib/ src/ .git CMakeLists.txt Makefile meson.build configure setup.py` 之一时才接受它，
否则**报错退出**并给出 `--project` / `--source-root` 的修法；`pdg_<project>.json` 缺失同样报错。
历史事故：曾把无法解析的 faiss 报告静默换成 `project/aria2`，跑出一个"看起来合理"的错误结果。
`main()` 在**第一次 LLM 调用之前**就预解析全部报告，误调用在秒级失败而不是几十分钟后的 ERROR 行。

### 阶段 2 — 路径分段

**做什么**：按报告路径上的控制事件把 `bug_path` 切成 `SegmentInfo`（每段 = 一个函数的连续片段）。

**关键约束**：`SegmentInfo` 只带**裸方法名**，所以段的函数归属必须按 **(file, 行区间)** 解析，
不能按名。按名会撞同名函数（`search` → `IndexPQ::search` 而非 `MultiIndexQuantizer::search`）。
`CFGCacheSet` 按 `segment.function_file` **按需加载并合并**多个 TU 的 CFG——
只加载单个 TU 会静默丢段。

### 阶段 3 — PDG 反向切片

**做什么**：报告事件 → 种子节点 → 沿 SDG 反向切片（跨函数），产出 `SliceMask`：
**哪些 PDG 节点是"与根因相关"的**。阶段 8 的路径空间压缩完全依赖它。

**关键约束**：切片深度预算**不跨函数累计**——函数内的 `control_dep` 边曾吃掉整条预算，
导致 `truncated=True`、相关性判定退化为"全保守"、bound 爆炸到 10^25。
`SliceResult.truncated` 会一路上传，并在阶段 8/9 的"推广"判定上作为 soundness 前置条件。

### 阶段 4 — 假设可行性

**做什么**：把 CSA 报告每一步自带的假设（`Assuming the condition is true/false`、
`Value stored to 'x'` 等）翻译成两层约束：

- `hard_constraints` — 硬约束（不可违反）
- `cumulative_state` — 变量→值的状态绑定

三层检查：规则层 → LLM 层 → 合并层。产出的 `cumulative_state` 在阶段 9 作为额外的
`extra_preconditions` 注入选路 prompt。

**位置**：跑在分支过滤**之前**（因为阶段 5 要用它作为"已建立状态"的种子）。

### 阶段 5 — 分支过滤

**做什么**：对每段从 CFG 抽取**可达**的分支，逐个做 4 层冲突检查：

| 层 | 判据 |
|---|---|
| 1 | 规则层：谓词与硬约束字面矛盾 |
| 2 | LLM 层：语义矛盾 |
| 3 | CSA postcondition 确认（`ReturnValueMapper`） |
| 4 | 合并层：与 `established_state` 矛盾 |

判定三态：**`consistent`**（可行）/ **`contradictory`**（确定不可行）/ **`insufficient`**（证据不足，
保守保留）。`insufficient` 是**默认态**——绝不因为"没证据"就剪枝。

**结构剪枝 A/B/C**（保守，只做减法）：

- **(A) 可达性**：按 CFG 从 `entry_block_id` 走 `successors[0]=true / [1]=false`，剪掉段内**未走**的
  `if` 臂。该顺序是 Clang 约定且**经过验证**（三元表达式实测为 `[then, else]`）。
  但 A 依赖 CFG block id 与 PDG node `block_id` 的配对，而两者来自**不同 frontend**
  （clang `-cc1 -analyze` 会插入隐式/临时析构/`do..while` scope 块，PDGBuilder 的 `CFG::buildCFG`
  默认 `BuildOptions` 不插），所以 id 会整体错位。`_alignment_score` 因此按**内容**比对两套编号
  （≥5 个块可比且 ≥0.7 一致才启用 A）。faiss 上只有小函数达标，**`read_index` 得 0.17 ⇒ A 实际不生效**，
  当前由 B/C 承担剪枝。要让 A 生效必须让 block 图从 PDG 侧产出，**不能靠重新信任 id**。
- **(B) 切片保留**：超出 `shallow_callee_depth`（默认 1）的 callee，只有
  `slice_mask.pdg_retained_nodes` 保留了它才展开。
- **(C) 结构护栏**：展开栈（杀掉 `read_index` ↔ `read_ivf_header` 互递归）；`source_root` 之外的 callee
  不递归（其谓词保留——仍能构成矛盾）；`_MAX_SERIALIZED_DECISIONS` 限制写进结果 JSON 的候选分支数。

**leaf-sentinel 契约**（改动会静默打断冲突检查）：未展开的 callee 仍留一个节点，
`callee_name` 有值、`condition_expr == "→ name()"`、`node_id == "callee_<caller>_<callee>_L<line>"`。
`shortest_path`、`ConflictChecker`、`branch_relevance._parse_callee`、`_normalize_call_edge` 都读这个形状。

---

## 3. "枚举前返回 FP"的出口

这三条是本方法**最便宜也最可靠**的 FP 结论：在路径选择与 POC 之前就返回，
不枚举那个可能上亿的路径空间。阶段 6 的两条（6a/6b）是**纯确定性**的
（不需要模型，只需报告 + 源码），阶段 7 需要一次 LLM 调用。

### 阶段 6a — 报告路径自相矛盾（确定性，无 LLM）

**原理**：CSA 每一步只印方向（`Assuming the condition is true/false`），**不印值**；
条件的**文本**只存在于 CFG/PDG 分支树里（按源码行索引）。把
「报告声称的方向」与「该行的条件文本」按 (行, 列) 配对成 **(条件, 方向)**，若**同一条件**在
报告**自己的路径**上被判成**两个方向**，且其间无写入 ⇒ 没有任何输入能实现该路径 ⇒ **FP**。

**为什么 CSA 自己推不出来**：当操作数来自**跨 TU 的函数**时（如 `fourcc` 在
`faiss/impl/io.cpp`），两次调用结果对求解器是**独立的自由符号**，无法互相约束。

**为什么既有三层一致性检查都看不见**（这是当初它漏掉的原因）：
报告步骤文本不含值 ⇒ `_check_constraint_conflict` 无从比较；
4 层 `ConflictChecker` **从不比较分支与分支**；LLM 合并检查判的是**被选中的组合**，不是报告路径。

**实测**（faiss 13 份报告只命中 2 份，都是 ground-truth FP）：
- `read_index`-484-1：`h == fourcc("IHNs")` 在 L901 判 **FALSE**（事件 #77）、在 L908 判 **TRUE**（事件 #84）；`h` 只在 L506 被写过。
- `read_index`-484-2：`h == fourcc("INSp")` 在 L921（`INSf || INSp || INSs` 短路链内）判 FALSE、在 L925（该臂内的 `if`）判 TRUE。

TP 的 `clone_index` 与其余十份 FP 均不命中。

**五道 soundness 护栏（全部 fail-closed，缺一即不判定）**：

1. 同函数 + 同文件；
2. 不同源码行（同一行的二次出现可能是循环重入）；
3. 其间**无写入**——源码扫描 *与* 报告自身的 `stored to 'x'` 事件都要干净；源码不可读 ⇒ 不产生 claim；
4. 其间**无 call step**——未建模的调用会把全局量重建成新符号（递归是特例），
   故**任何** call step 都否决该配对；
5. 行内配对必须**无歧义**（该行的 PDG 节点数必须等于报告该行的步数；
   `||` 短路产生**少于**节点的步数 ⇒ 计数不符 ⇒ 跳过）。

**出口**：`verdict="FP"`，结果 JSON 带 `report_self_contradiction` 块（条件、变量、两处出现各自的
行/方向/事件号），**无 `poc_result`**，在阶段 8 之前返回。

#### 条件同一性的两段历史（均已修，已实测）

faiss 的两次命中（484-1/484-2）是有效的：条件是 19 字符的真谓词 `h == fourcc("IHNs")`。

protobuf 21 份的批次实验里曾出现 **5 次命中没有一次建立在有效同一性上**——3 次把 ground-truth
**TP** 判成 FP（`subprocess.cc-Start-{28,31,58}-1`），另 2 次答案虽是 FP 但理由同样无效
（`stringprintf.cc-StringPrintfVector-24-1`、`time_test.cc-CmpHelperEQ-1-1`）。两个成因，**都已从
源头修掉，不需要 Python 侧护栏**：

1. **陈旧产物**（`pdg_protobuf.json` 声明 1.0，是旧的 `build/pdg_builder_local` 二进制产出的，
   而源码已是 1.2）。同一份 compile db 重建 1.2 后：控制依赖边 23,746 → 33,729（+42%），分支谓词
   从宏记账节点回到真谓词（`v.size()` → `!(!((v.size()) <= (kStringPrintfVectorMaxArgs)))`，
   `pipe(...) != -1` 的 `event` 节点 → 真正的 `if_cond`）。版本已加三道闸门，见
   [`docs/dependency-graph-build-guide.md`](dependency-graph-build-guide.md) §3.1。
2. **表达式截断**（`PDGBuilder.cpp` 的 `> 120` → `substr(0,117) + "..."`，3 处）。重建 1.2 之后
   仍留下 `time_test.cc-CmpHelperEQ-1-1` 的 6 次命中：L62/L64/L65（判 FALSE）与 L66/L67（判 TRUE）
   其实是五条**不同**的 `EXPECT_EQ`（`1970==time.year`…`1==time.second`），但 gtest 断言宏的失败臂
   调用 `AssertHelper(...)`，截断后五条逐字节相同。**v1.3 去掉截断后它们变回 10 个互不相同的串**
   （各自带 `62/64/65/66/67`），假同一性无法再形成；顺带修好了引号被切半导致 `_STRING_LIT_RE` 失效、
   `_condition_variables` 返回路径碎片（`home`、`project`…）从而让护栏 3 形同放行的问题。

**实测基线（`tools/probe_report_consistency.py`，v1.3 产物）**：protobuf 21 份**全部 0 命中**；
faiss 4 份不变——484-1 / 484-2 各 1 处（claims 55 / 56），`fourcc_inv_printable-5-1` 与
`clone_index-2-1` 各 0 处。任何后续改动都必须让这两组数字保持不变。

### 阶段 6b — 入口状态制造：穿过必然终止的检查（确定性，无 LLM）

`entry_state.detect_fabricated_entry_state`，与 6 共用一个早返回出口。

**原理**：报告的**第一步**是 `Assuming pointer value is null`——CSA 把该函数当成自己的入口，
假设形参进来就是 null。若**该函数自己的函数体**在这条解引用之前带一个**无条件编译**的致命检查
（`ABSL_RAW_CHECK` / `ABSL_INTERNAL_CHECK` / `ABSL_CHECK` / `GOOGLE_CHECK` / `CHECK` /
`FAISS_THROW_IF_NOT{,_FMT,_MSG}`），条件要求这个形参**非 null**，而**报告自己的路径**又记录了在该行
取得 **TRUE 分支**（对 `if (!(cond)) FATAL` 家族，TRUE 就是**失败臂**），则：失败臂**不返回**
（absl 的 FATAL 处理器落到 `raw_logging.cc:183` 的 `abort()`；`FAISS_THROW_IF_NOT` 抛异常），
于是「检查失败」与「继续解引用」**不可兼得** ⇒ 该路径不可实现 ⇒ **FP**。

这条正是阶段 10「POC 自造前提」缺陷的**确定性一半**：见 §6 的说明。

**为什么既有的三层都看不见**：

- 模拟审计的"制造入口"规则是按 **POC 调用了哪个函数**分档的（内部/叶子/template helper、
  被 stub 的库函数、丢掉了 override 的子类）。当被标记的函数**本身就是一个公开 API**（absl 的
  `SimpleAtob`）时，POC 直接调用它正是提示词要求做的，一条规则都不匹配；而提示词里既没有调用方
  证据、也没有被调函数的函数体，审计**没有证据**说明这个 null 无法到达。
- `CHECK` 家族的**宏内部**不是 CFG/PDG 能比较的分支条件（它展开成一条**语句**，不是谓词对），
  所以阶段 6 的同条件反向检测也看不见。

**六道 soundness 护栏（全部 fail-closed，缺一即不判定）**：

1. bug 类型必须含 `null`，且入口 null 假设是报告的**第 1（容忍 `Error Start` 在前时第 2）**个事件；
2. 被解引用的变量必须取自报告**最后一个 `report` 事件**的 `loaded from variable 'V'`，且必须是
   被标记函数的**指针形参**（成员、局部变量、无法解析的签名一律不判定）；错极性（`V == nullptr`）
   与复合条件（`V != nullptr && x`）一律不判定——`!V` 也**不算**（它只在 V 为 null 时成立，
   与报告假设一致而非反驳），只有偶数个 `!` 才剥；
3. 检查必须来自**无条件编译**的家族：`assert` / `DCHECK` / `FAISS_ASSERT` **故意排除**——它们
   在 NDEBUG 下被编译掉，这正是 CSA 的 null 得以存活的原因，据其判 FP 是不成立的；
4. 检查必须在**解引用之前**、且在**同一函数体内**（函数范围由签名 + 花括号体求出，多行/宏生成的
   签名直接放弃）；
5. 宏调用必须**在本行闭合**（多行 `CHECK(a,\n b)` 放弃）且首个实参可证"要求非 null"
   （`V != nullptr` / `NULL != V` / `V != 0` / `!!V` / 裸 `V`）；`CHECK_EQ` 这类**双操作数**
   形式由 `(?![_A-Z])` 排除——"要求非 null"无法从首个实参表达，宁可不判；
6. 报告**自己**必须在该行记录 `branch_condition is True`。

**调用点扫描只是旁证，永不单独定论**：`scan_call_sites` 统计该函数在源码树里的真实调用点数
（跳过注释行、`build/`、`.git` 等，有 40k 文件 / 512 MB 上限；超限返回 `None` = **未知**，绝不当作 0），
结果只写进 reason（"共 0 个真实调用点"）。公开库 API 的调用方可能在语料之外，**"无树内调用方"**
单独不足以判 FP——定论只来自上述 6 条（全部局部、可在报告与 bug 文件内验证）。

**实测**（`tools/probe_report_consistency.py` 的 `entry-state` 段）：protobuf 21 份命中 **2** 份
——`numbers.cc-SimpleAtob-11-1` 与 `-5-1`（形参 `out`，`ABSL_RAW_CHECK(out != nullptr, …)` @
`numbers.cc:109`，1.0 秒内）；faiss 13 / aria2 11 / folly 8 份**全部 0 命中**（含最尖锐的回归样本
`clone_index`——那里的 null 来自真实调用方的数据通路，按用户定标属 TP，不得命中）。

### 阶段 7 — 语义有限域冲突（LLM，枚举前）

**原理**：CSA（以及工具自己的约束模型）会把某些变量当成可任意取值，而源码其实把它钉在一个很小的常量集上
（典型：文件头判别符 `h` 与跨 TU 的常量工厂 `fourcc("...")` 比较）。
**domain supplement**（`extract_domain_facts`）先扫出这类
"guard-pinned / 有限域"变量（被 guard 钉住、折叠出有限个常量），作为**建议性事实文本**
注入选路 prompt 与模拟器 prompt（Phase 1，仅建议性质）。

**`semantic_fp_reason`**（Phase 2，决定性）：在**枚举之前**单发一次 LLM 调用，问：
报告路径是否把某个有限域变量逼进了自相矛盾（例如：只有 `h ∈ S` 才被允许进入解引用区域，
而所有会给指针赋值的 dispatch 臂都取了 FALSE ⇒ 指针恒为 null）。
判定为是 ⇒ **FP**，在路径选择前返回（`semantic_feasibility.decisive_fp`）。

**保守性**：单次调用；任何异常或非决定性回答都不判定，继续走正常流程——闸门永不致命。

**实测**：`read_VectorTransform` 由 UNKNOWN → 决定性 FP；反例安全性由 `operator=`（TP）与
leak `484-2`（FP）保持不翻转来验证。

---

## 4. 阶段 8 — 路径空间收集与压缩（纯确定性）

**做什么**：把阶段 5 产出的每段可行候选求**整路径叉积**得到 `whole_path_bound`，
然后逐层压缩：

1. **`classify_branch_relevance`** — 用 SDG 切片把每个枚举分支标成
   **relevant / irrelevant**（谓词到 bug 触发点有无数据/控制通路）。irrelevant 的分支**固定**，不参与枚举。
   实测 SocketAddress 75% 分支无关，9216 → 16。
2. **`reduce_path_space`** — 按相关性压缩，得到 `reduced_bound`。
3. **`annotate_ci_groups`** — **上下文不敏感折叠**：多个段若携带**完全相同**的分支集
   （同一函数的分支在多段重复出现，如某循环的 4 个段），则它们的方向被**绑在一起**（全路径一致），
   叉积从 ∏counts 塌成一个分支集（GlogFormatter 4096 → 8）。
   **只在真正的重复调用点（`callee_`/`call_`）折叠；循环重入（`seg_`/`pdg_`）绝不折叠**——那曾不健全地把
   GlogFormatter 从 TP 翻成 FP。
4. **`iter_combinations`** — 提供整路径组合枚举器（有枚举上限），供阶段 9 逐组合推进。

> **bound 只反映压缩后的相关空间**，且受每段候选 **cap（`_MAX_CANDIDATES`）** 截断——
> **跨运行比较 bound 前必须先固定 cap**，否则比的是两个截断位置。

**探索预算**：`reduced_bound ≤ _FULL_EXPLORE_THRESHOLD (32)` ⇒ **全空间探索**（尝试次数 = bound）；
否则退回到 `max_path_attempts` 的有界采样。这个区分决定了阶段 9 能否下 FP 结论（见下）。

---

## 5. 阶段 9 — 路径选择

**做什么**：冲突驱动的**整路径组合枚举**。第 1 次用 LLM 选"最优可行路径"；
之后每次**屏蔽整个整路径组合**（绝不屏蔽单段分支——不可行性是组合相关的：
`(segA=b1, segB=c1)` 是 FP 不代表 `b1` 不可行），再枚举下一个不同组合。

**为什么屏蔽的是"整路径组合"**：阶段 10 对一条路径判 FP 只说明**那条**路径不可行，
不能证明报告是 FP——其它组合可能真的能到达 bug。

阶段 9 的出口（按优先级）：

| 情形 | 结论 |
|---|---|
| LLM 找到全局一致可行路径 | 交给阶段 10 |
| 该组合**全局不可行**（`combo_infeasible`） | 屏蔽该组合，枚举下一个（**不是**报告级 FP） |
| **真·冲突式不可行**（冲突检查器/卡死，无任何全局一致路径） | **FP**（sound） |
| 枚举器耗尽 / 穷尽且无新组合 | 转交阶段 9 末尾的聚合 |
| 模拟执行判 FP + `reduced_bound == 1` | **路径空间压缩推广 → FP** |
| 模拟执行判 FP + 结构性矛盾 + 根因指针不变 | **结构性FP根因不变性推广 → FP** |
| 单路径 FP（其它情形） | 屏蔽该路径继续探索（**单路径永不单独定案**） |
| 模拟执行 unresolved | 屏蔽该路径，重选 |
| 验证出错（环境问题，非分类结论） | 保留该路径为 **TP（未证实）** |

### 阶段 9 末尾的聚合（Tier 2）

| 情形 | 结论 |
|---|---|
| leak 层跨路径**一致**判 FP（≥2 个见证，且无 TP 见证） | **leak 层一致性推广 → FP** |
| 全空间探索 + 全部路径均为决定性 FP + 无 unresolved | **路径探索聚合 → FP** |
| 大空间的有界采样（`reduced_bound > 阈值`） | **UNKNOWN** |
| 探索中存在 unresolved | **UNKNOWN** |
| leak 层跨路径**自相矛盾** | leak 层证据作废，落到其它判据 |

**四条"推广"的证据基础**（都不是文本匹配，都是图/结构证据）：

1. **路径空间压缩推广**（`reduced_bound == 1`）——压缩后只剩**一个根因等价类**，
   枚举到的分支全部经切片确认与根因无关 ⇒ 没有别的组合能改变结论。
2. **结构性FP根因不变性推广**——决定性结构性 FP（`csa_contradiction` 或 LLM 标 `definite_fp`）
   是关于**根因指针**的报告级断言；若相关空间里**没有任何相关分支**改变该指针的方向，
   则该矛盾在每个组合上都成立。前置：结构性 FP + 成功取回指针 + 切片未截断。
3. **leak 层一致性推广**——leak 层判的是**对象的 ownership chain**（谁最终持有它、
   是否为 abandon 需要越界输入），这是关于 **callee 代码**的断言、已对整个空间普遍成立，
   且枚举分支只改变"读哪种索引类型"、不改变该对象的归属。故**不需要空间穷尽**，
   但需要多次尝试**一致**（leak chain 每次重读，484-1 曾 FP/FP/TP ——单次定案会让结论取决于尝试顺序）。
4. **路径探索聚合**——压缩相关空间已**穷尽**，全部探索路径均为决定性 FP。
   **仅在全空间探索时成立**：对巨大空间的有界采样，即使全部 FP 也不能证明 FP（未试的组合可能有解）。

---

## 6. 阶段 10 — POC 验证与结论

对阶段 9 交来的每一条可行 `selections` 跑一遍（`_verify_path`）：

| 子步 | 做什么 |
|---|---|
| 10a | 抽取源码上下文（`FPAnalyzer.analyze`） |
| 10b | **约束补全** → `CompletedPath`（preconditions + 具体化取值），**锚定所选路径**，不同选择产生不同 POC |
| 10c | **生成 POC**（纯 LLM，**不真编译**）→ 写 `poc_<stem>.cpp` |
| 10d | **LLM 模拟执行 + 反馈循环**（≤3 轮）：`SimulationVerifier.verify_with_feedback` |
| 10e | 返回 `conclusion ∈ {tp, fp, unresolved, None}`，**不改判定**——由阶段 9 的循环决定 |

**模拟审计是防幻觉的核心**，两条硬约束：

- **逐步路径一致性**（`path_conformance`）：POC 必须复现 CSA 报告路径的**每一步**，
  否则该轮不接受（回灌重生成），**triggered 也不接受 TP**。
- **绝不因解析失败翻 FP**：回复被截断/需修复时标 `parse_degraded`，
  强制 `path_conformance_ok=False` ⇒ 该轮重生成；退化为 **UNRESOLVED 而非 FP**。
  （历史事故：截断的回复被正则抠出 `triggered: true`、`path_conformance_ok` 默认 `True`，
  静默解除幻觉闸门，两次假 TP 都来自这样的轮次。）
- **稀疏审计格式**：模拟回复是全线最长最贵的一次调用。审计只报
  `steps_checked` + `path_mismatches` 列表（未列出即视为满足），
  由 `_expand_sparse_audit` 还原成完整 `path_conformance`。
  `steps_checked` **必须等于**渲染的步数（局部审计不能伪装成通过）；
  `False` 配空列表、重复/越界步号、计数不符 ⇒ `parse_degraded` ⇒ 重生成。
  曾试过"位串"与"满足计数"两种更省的聚合，**都因格式差错丢掉决定性轮次而回退**——不要重新引入。
- `SimulationVerifier._csa_step_rows` 是步号**唯一来源**（渲染的清单与回复的索引编号绝不能各自独立）。

**leak 层**：内存泄漏类报告额外走一条 `leak_chain`（alloc → abandon → unowned）判定层，
**仅损坏输入才能触发的泄漏判 FP**。

#### "POC 自造前提"缺陷与四层修复（阶段 6b 的由来）

阶段 10 原来问的是**"有没有代码能让 CSA 的前提出现在那里？"**，而不是**"真实调用方能到达这里吗？"**——
于是**POC 自己把病理入口值传进去**也算"触发"。protobuf 的 `numbers.cc-SimpleAtob-{11,5}-1`
（ground-truth FP）正是这样拿到 TP 的：POC 直接 `SimpleAtob(str, nullptr)`，
而函数体 L109 的 `ABSL_RAW_CHECK(out != nullptr, …)` 是无条件编译的致命检查、失败即 `abort()`。
四层修复（每层都可独立生效，都不加模式开关）：

| 层 | 位置 | 做什么 |
|---|---|---|
| F1 | 阶段 6b（`entry_state.py`） | **确定性**判定：入口 null 假设 + 报告自己穿过必然终止的检查 ⇒ 枚举前 FP |
| F2 | `source_context.extract_callee_sources` | 把报告 `Calling 'X'` 事件的**真实函数体**同时喂给 10c 生成器与 10d 审计器（原来两边都只看到 bug 文件，`time.cc` 等**异名文件**里的被调函数对审计器不可见——这正是"输出参数未写入"的错误断言得以存活的原因） |
| F3 | `prompt_templates` | 生成提示词新增禁用捷径 (e)（按**取值来源**判：不能靠"自造入口状态"让前提成立）；规则 9 改为把 `// POC_UNREACHABLE: <理由>` 写在**首行**；模拟提示词新增"入口状态制造"条款与**调用结果的 ASSUMPTION** 条款（报告假设某次调用"返回但未写输出参数"时必须对照真实函数体验证） |
| F4 | `simulation_verifier._is_manufactured_trigger` | 审计回复新增 `bug_value_source`（`poc_argument` / `caller_data` / `api_return` / `unknown`）：当**触发值只能来自 POC 自己的实参**时，该轮 `triggered` 覆写为 false 并判 **FP**（`poc_argument` + `triggered` 同时出现才算；其余取值一律不覆写） |

**生成器自我声明的通道**（F3 的消费者）：`// POC_UNREACHABLE:` 首行由
`prompt_templates.parse_poc_unreachable_note` 解析后，作为**"待裁决的主张"**注入审计提示词
（只在该 POC 自己的第 1 轮生效，精化后的 POC 另行判定）。它**不能**单独定案：
审计器必须自己复现该理由才可判 FP，措辞里明写"主张而未复现的理由不是 FP 的依据、更不是 `triggered` 的依据"。
（历史：这条出口此前**没有消费者**——生成器明知不可达也必须假装可达。）

---

## 7. verdict 总表

| verdict | 可由哪些阶段得出 |
|---|---|
| **FP** | 阶段 6a（报告自相矛盾）、阶段 6b（入口状态制造）、阶段 7（语义有限域冲突）、阶段 10d（`bug_value_source == poc_argument` 且 `triggered`）、阶段 9（真·冲突式不可行）、阶段 9 的四条推广（压缩/结构性/leak 一致性/探索聚合） |
| **TP** | 阶段 9：模拟执行判定触发 bug；leak 层一致判 TP；验证出错时保留为 TP（未证实） |
| **UNKNOWN** | 阶段 9 聚合：巨大空间的有界采样、探索中存在 unresolved |

---

## 8. 设计不变量（防回归，勿随重构删除）

1. **无模式开关**。冲突驱动选路是唯一的选路机制，POC 验证恒开。历史上的
   `--verify`/`--no-verify`/`--v2`/`FP_USE_CONFLICT_LEARNING` 全部已删——不要再加。
2. **阶段 5 的 `insufficient` 是默认保守态**，不是"不可行"。
3. **A/B/C 是只做减法的保守规则**；A 的 block id 配对必须靠 `_alignment_score` 的内容比对，
   不许重新信任 id；`_MIN_ALIGNED_BLOCKS=5` / `_MIN_ALIGNMENT=0.7` 未达标即停用 A。
4. **leaf-sentinel 契约**（`condition_expr == "→ name()"` + `node_id == "callee_<caller>_<callee>_L<line>"`）
   被四处消费者依赖；gated callee 无谓词无调用时**不发节点**，所以门控只能删节点、不能加。
5. **单路径 FP 永不单独定案**；FP 必须来自"冲突式不可行"或"全空间穷尽/推广"。
6. **模拟审计**：稀疏格式 + `steps_checked` 必须等于渲染步数；`parse_degraded` ⇒ 重生成、
   退化为 UNRESOLVED，**绝不翻 FP**。
7. **渲染上限**（`_MAX_RENDERED_DECISIONS`/`_MAX_RENDERED_STATE_CHARS`、
   `path_space._MAX_SERIALIZED_DECISIONS`）不可移除——真实路径空间可达数万分支/段，
   `established_state` 可达 MB 级。注意 `signature` 与 `ci_branch_keys` 用的是**完整**分支列表，
   只有写进结果 JSON 的那份被截断。
8. **默认 max_tokens 是 8192**（`_SIMULATION_MAX_TOKENS` 用于模拟/精修调用）；
   `finish_reason` 为 `length`/`max_tokens` 时 `_call_llm` 必须告警。
9. **阶段 1 的路径解析必须 fail-fast**：绝不替换成别的 project（见阶段 1 的历史事故）。
10. **产物存全量、只在渲染时截断**。`pdg_<project>.json` 里的 `expression` 是完整文本
    （v1.3 起不再有 117 字符 cap）；需要短文本的消费者——prompt、结果 JSON——必须在**渲染**
    时截断，绝不靠存一份被截断的产物。反例就是阶段 6 的那次误判：截断把 gtest 断言宏的五条
    不同 `EXPECT_EQ` 压成同一段文本，"同一条件"于是被凭空建立。
11. **阶段 6b 的致命检查家族只收无条件编译的**：`assert`/`DCHECK`/`FAISS_ASSERT` 在 NDEBUG 下
    被编译掉（那正是 CSA 的 null 得以存活的原因），据其判 FP 不成立——不要再往里加；
    `CHECK_EQ` 这类**双操作数**形式也故意不判（首实参表达不了"要求非 null"）。
    **调用点扫描永远只是旁证**（写进 reason），"树内无调用方"单独不足以判 FP，也不许升级成独立判据
    （公开库 API 的调用方可能在语料之外）。`_requires_non_null` 对 `!V`（奇数个 `!`）必须返回 False。
12. **`bug_value_source` 的覆写是单向的**：只有 `poc_argument` **且** `triggered` 同时成立才覆写为
    `triggered=False`/FP；其余取值（`caller_data`/`api_return`/`unknown`/缺失）一律不覆写。
    这条不变量保护的是 `clone_index` 那类样本——null 来自**真实调用方的数据通路**时属 TP，
    不得因为"值最终来自某个调用"就被判 FP。
13. **被调函数体是"证据"不是"结论"**：`extract_callee_sources` 解析不出来时只输出空块，
    由审计规则自行判断；生成器的 `POC_UNREACHABLE` 主张同样只作为**待裁决的主张**注入，
    审计器必须自己复现才可判 FP，也不得据其判 TP。

---

## 9. 复现与回归工具

`tools/` 下的四个脚本（原在 `output/`）：

| 脚本 | 用途 |
|---|---|
| `tools/replay_steps1to6.py` | 重放确定性阶段 1–6（无 LLM），打印每段的 fn/file/CFG key/分支数。**每份报告新建 `CFGCacheSet`**，避免一份报告的 TU 影响另一份。 |
| `tools/abc_compare.py` | 同一批报告跑两遍（A/B/C 关 vs 开），对比每段节点/分支数与整路径 bound。 |
| `tools/probe_report_consistency.py` | 确定性阶段 1–6 + **阶段 6 的两条闸门**（6b 入口状态制造 + 6a 矛盾检测），打印入口状态判定、claim 直方图与每处矛盾。6b 只读报告与源码，故不依赖 PDG/CFG cache。 |
| `tools/summarize_faiss_eval.py` | 汇总 `output/` 下的结果 JSON，按 ground truth 打分。 |

**回归集**：`484-2`、`484-1`、`clone_index`（必须保持 2 段 / 4 分支 / bound 16）、
`fourcc`、`IndexFastScan`、`read_VectorTransform`、`test_merge`。

```bash
PYTHONPATH=src python3.10 -m pytest tests/ -q
python3.10 tools/probe_report_consistency.py <484-1> <484-2> <fourcc> <clone_index>
python3.10 tools/replay_steps1to6.py <回归集>
python3.10 tools/abc_compare.py      <回归集>
```

**阶段 6 的两条基线**（v1.3 产物上实测，任何改动都必须让它们保持不变）：

| 语料 | 期望 |
|---|---|
| faiss 4 份 | `484-1` / `484-2` 各 **1** 处矛盾（claims 55 / 56），`fourcc_inv_printable-5-1` / `clone_index-2-1` 各 **0** |
| protobuf 21 份 | **全部 0**（含曾误报的 `subprocess.cc-Start-{28,31,58}-1` 与 `time_test.cc-CmpHelperEQ-1-1`） |

**阶段 6b 基线**（不需要 `--project` 的 PDG 数据也能跑；同一 `probe` 输出里的 `entry-state` 段）：

| 语料 | 期望 |
|---|---|
| protobuf 21 份 | **2** 份命中：`numbers.cc-SimpleAtob-11-1` 与 `-5-1`（形参 `out`，`ABSL_RAW_CHECK`，调用点 0） |
| faiss 13 / aria2 11 / folly 8 份 | **全部 0**（含 `clone_index`——那里的 null 来自真实调用方数据通路，不得命中） |

判定阶段 1–9 的计数基线（v1.2 = v1.3 逐项相同）：`IndexFastScan` 44 分支、`fourcc` 1152、
`484-1` 34 段 10152 分支、`484-2` 45 段 10136 分支、`read_VectorTransform` 119、
`test_merge` 34、`clone_index` 2 段 4 分支 bound 16；A/B/C 在 faiss 这 7 份上是 no-op。

评估脚本按 **`verdict_step` 里的中文关键字**打标记（如 `"遍历超限" in row["step"]` 标
`traversal-limit`）——**改阶段编号时必须保留这些关键字短语**。
