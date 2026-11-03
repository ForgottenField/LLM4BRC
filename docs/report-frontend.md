# 报告处理前端（report frontend）

本文描述 `src/report_frontend/` 这一层：它把**任意一种静态分析器的错误报告**读成同一个
模型，并记录报告所指的**代码**。`docs/pipeline-stages.md` 是分类方法的权威描述，本文是它的**前置
一章**——阶段 1 的第一件事现在就是调用这一层。

> 结论先写在这里：**适配器只负责"把报告读成模型"，模型只负责"报告说了什么 + 它指的是哪段代码"。
> 任何"让非 CSA 报告看起来像 CSA 报告"的转换都已经被证明是错的（见 §1）。**

---

## 0. 模块地图

| 层 | 文件 | 职责 |
|---|---|---|
| 规范模型 | `models.py` | `BugFinding`（断言了什么）、`TraceStep`（怎么走到那里）、`SourceLocation`（在哪）、`ReportDocument`（一个文件里的全部 finding） |
| 代码模型 | `code_model.py` | `SourceCodeModel` / `ResolvedFile`：把报告里的路径解析到本地源码树，并**记录用了哪一级策略**；报告自带源码时从报告读 |
| 适配器 | `adapters/csa_html.py`、`adapters/sarif.py` | 每种容器格式一个文件，归一化到规范模型 |
| 注册与探测 | `registry.py` | 适配器注册表、`sniff()` 打分、歧义处理、`--report-format` 覆盖 |
| 门面 | `frontend.py` | `ReportFrontend.load()/expand()/iter_findings()`、`ReportRef`（`<path>::<finding>` 寻址）、`LoadedReport` |
| 错误 | `errors.py` | 失败一律抛这一族异常，绝不漏出 `json.JSONDecodeError` 之类 |
| 兼容层 | `fp_analysis/report_bridge.py` | 把规范模型**渲染回**流水线已有的 `ParsedReport`/`BugPath` |
| 内省 | `adapters/base.py` | `--list-formats` 的表格 |

约 2900 行（含适配器与测试驱动的注释），其中 CSA 适配器是**原解析器的 v2**：文档走法一字未改，
新增的是**事件分类**（`StepKind`）。

---

## 1. 为什么要重构

流水线原本只有一种输入（CSA HTML），于是"报告解析结果"**同时充当了流水线的内部表示**：一个
`ParsedReport` + 一串以 `path_number`/`event_kind`/`description` 为键的字典。这套表示里有三处是
CSA 独有的、且被下游十几个阶段解码的：

- `path_number == 9999`（CSA 的 `EndPath`）是**唯一**能识别"报告本身那一步"的信号；
- `Calling 'X'` 这种**自然语言措辞**是识别"进入被调函数"的唯一途径；
- `branch_condition` 只由 CSA 的 `← Taking true/false branch →` 措辞给出。

要接第二种分析器，只有两条路：让下游每个阶段都学第二套词汇，或者**把第二种报告转成 CSA 的
HTML 形状**。当时实际做的是后者——把 CodeQL 的 finding 手工改写成 CSA 报告，于是 frontend 的
门面文档里写着"这就是这个包存在的理由"：

- 转换是**有损且静默**的：SARIF 里根本没有分支方向，转换出来的"CSA 报告"也不会有；下游读不到，
  于是保守地停在 `insufficient`——这还算好；**坏的是有人给它补上一个值**，那就会凭空造出一次
  "条件冲突"或一次 `path_conformance` 通过；
- 转换是**不可维护**的：62 份 CodeQL finding 的 `--list-findings` 就是一次转换的产物，第三、
  四种分析器（Infer / Cppcheck）无法这样接；
- **两边都难以发现问题**：报告里的 62 个 finding 挤在一个文件里，寻址、输出命名、批处理全部按
  "一份报告 = 一个工作单元"写死了。

所以这一层要解决的是三件事：**一个模型**（报告说了什么）、**一套寻址**（多 finding 的文档怎么
指到其中一个）、**一次解析**（代码路径只解析一次、只按一套阶梯）。

---

## 2. 报告模型：只装分析器真的说过的话

`BugFinding` 是"一次断言"：`rule_id`/`rule_name`/`category`/`severity`/`message`、
`location`（bug 在哪）、`function`、`steps`（分析器给出的路径）、`reported_path_length`、
`related`、`fingerprint`、`raw`（适配器原文）。

`TraceStep.kind` 的词汇表是**所有分析器的并集**，刻意小、刻意不用 CSA 的名字：

| `StepKind` | 含义 | 谁给得出 |
|---|---|---|
| `ERROR_START` | 被标记的值/状态**产生**于此 | 只有明确这么说的分析器（CSA `Error Start`） |
| `EVENT` | 普通数据流/观察步 | 所有 |
| `CONTROL` | 分支决策（带 `direction: bool`） | 只有给出方向的（CSA） |
| `ASSUME` | 分析器在此施加的约束 | CSA 的 `Assuming …` |
| `CALL` / `RETURN` | 进出被调函数（`callee`） | CSA；SARIF 的 `[Return]` 归一为 `RETURN` |
| `NOTE` | 分析器注释，不属于路径 | 任意 |
| `REPORT` | finding 本身（路径最后一步） | 所有（适配器负责保证） |

**规范模型不装的东西，下游必须拿不到。** 这条是本层的核心纪律，有四个具体后果（前三条都写在
`adapters/sarif.py` 的模块注释里）：

1. SARIF 的步骤**永不**被分类成 `CONTROL`——那是数据流轨迹，不是分支决策；给它一个 `CONTROL`，
   阶段 6a 会拿到一个它从未被告知的方向，阶段 5 会拿到一个它无法检查的条件。
2. SARIF 的第一个 flow location **不**被改写成 `ERROR_START`——污染类查询里它是 **source**，
   不是同一个断言。
3. 报告没有函数名就是没有（SARIF 没有 `FUNCTIONNAME`），前端不猜；CLI 从 PDG 反推（见 §7）。
4. **报告没有路径就是没有路径**（`path_evidence`）。这是四种分析器差别最大的一处：CSA 走符号
   路径并打印每一步；CodeQL 常常只报**位置**（实语料 176 个 finding 里 127 个没有 `codeFlow`）。
   只报位置的 finding **仍然会渲染出一个报告步**，所以下游数步骤个数是分不出"一步路径"和
   "没有路径"的，而这个区别恰恰是阶段 2 要不要分段的依据。因此它是一个**类型化字段**：

   ```python
   class PathEvidence(str, Enum):
       TRACE = "trace"            # 分析器走了路径（有 codeFlow / CSA 的每一步）
       LOCATION_ONLY = "location" # 只给了位置和断言
   ```

   **由适配器给，消费者不许从 `len(steps)` 反推**：适配器可以看整个文档（分析器的原话在那里），
   下游只看字段。默认值是**更弱的那个**（`LOCATION_ONLY`）——没说过的话不能算说过。
   流水线拿到 `location` 时有两个消费点，方向和理由不同：
   **阶段 2 跳过路径分段**，直接把报告位置所在函数作为分析单元；
   **阶段 10 不作 POC 验证、直接判 UNKNOWN**——验证 TP 的仪器是"逐步对照报告路径"，
   没有报告路径就没有对照物，审计必然空过，`triggered` 会立在我们自己选的路径上
   （详见 `docs/pipeline-stages.md` §2/§6 与 §8 不变量 17）。

`TraceStep.raw` 保留适配器的原文片段。它为兼容层服务：CSA 步骤的 `legacy_branch_condition`
就是从 `raw` 取的，所以**重构后 CSA 的下游输入逐字节不变**。

---

## 3. 代码模型：解析一次，并记住"凭什么"

`ResolvedFile` = `(reported, local, strategy, candidates)`。解析阶梯（`_resolve_uncached`），
**每一级都是比上一级更弱的断言**：

| # | `strategy` | 依据 | 强/弱 |
|---|---|---|---|
| 0 | `embedded` | 报告自带该文件的源码（无需本地文件即可出 snippet） | — |
| 1 | `direct` | 报告里的绝对路径在本机就存在 | 强 |
| 2 | `build-prefix` | 剥掉已知构建根（`KNOWN_BUILD_PREFIXES`）后拼到本地根 | 强 |
| 3 | `relative` | 相对路径（SARIF / Cppcheck 的 `uri`）相对源码根，或相对文档给的 URI base | 强 |
| 4 | `suffix` | **最长后缀**在树中存在——钉住了位置，扛得住任意前缀和不同名的顶层目录 | 强 |
| 5 | `basename` | 只剩文件名。**唯一可能歧义的一级** | 弱 |
| 6 | `unresolved` | 没有 | — |

`STRONG_STRATEGIES` 故意**不含** `basename`/`embedded`/`unresolved`。歧义只可能出现在第 5 级：
`ResolvedFile.candidates` 记下同名文件（**选中的排第一**），`is_ambiguous`/`rivals` 供消费者判断。
`rivals` 是把选中项去掉后的列表——报告歧义时必须用它，否则消息会写成 `X → X (also: X, Y)`，
字面反过来。

> `basename` 那一级为什么必须存在又必须被怀疑：protobuf/abseil 里 `numbers.cc`、`time.cc`
> 各有好几份，`rglob(name)[0]` 是字母序巧合。`suffix` 之所以算"强"，是因为它钉住了路径位置。

**调用方按用途取用：要么接受强证据，要么接受"不知道"。** `shared_model(root)` 是给
"一个 root 上解析几万次"的调用方准备的单例（管线每条路径事件、每个 SDG 函数各解析一次）；
`fp_analysis/cfg_feasibility._resolve_local_path` 是它在本管线里的消费者，**只接受
`STRONG_STRATEGIES`**，弱命中一律返回 `None`。理由是这两者的下游不同：拿到 `None` 时事件会走
按行号匹配的兜底，而那条兜底**被限制在"报告自己提到的文件"里**；接受一个 `basename` 猜测则等于
把 `/usr/include/fmt/core.h` 断言成树里某个无关的 `core.h`，并直接把它当权威用于函数归因。
**错的文件比没有文件更坏。**

同一份代码模型也负责**读代码**：`snippet(file, line, radius)`、`read_range`、
`add_embedded`（适配器把报告自带的源码登记进来）。snippet 的取值顺序是**报告自带优先**，
其次才读本地树——报告自带的那份是分析器实际跑过的 revision。

**一个文件一份自带副本，按「份」而非按「行」取胜。** CSA 适配器给每个 finding 两份 embedded：
`raw["source_snippets"]`（整篇文档按**行号**过滤的窗口）与 `extract_file_lines` 的
`embedded_files`（**逐行归属到它自己的节**）。前者会把别的文件节的同号行收进来，
在 bug 文件该行是空行时更是直接**填进邻文件的代码**——实测 211 份语料里
**1828 行 / 62 份报告**，每一行都证实不是该文件在该行的内容，其中 **25 份报告的 bug 行本身**
就是别的文件的代码。所以 `frontend._register_embedded` 的规则是：文档列过的文件**只**要它自己那份
节副本，`source_snippets` 只登记给文档**没有**列出、因而不存在逐行归属的文件。
不能把两份 `add_embedded` 到同一个键上——合并只能**覆盖**，删不掉外来行（bug 文件是空行的位置
外来行会永久留下）。这也是 `report_bridge.source_snippets` 改走 `loaded.code.snippet()` 的原因。

**两份不一致就报（不裁决）。** `snippet()` 命中自带副本时会同时读本地同行，不一致记一条
drift：`{reported, line, embedded, local, strategy}`——`strategy` 是**怎么找到那个本地文件**的，
因为对着 `basename` 命中的不一致可能是「同名另一个文件」而不是「本地检出过期」，两者该采取的动作不同。
出口是 `logger.warning` + 结果 JSON 的 `code_model: {drift: [...], drift_truncated: N}`
（每条 `(reported, line)` 只记一次，上限 `_MAX_DRIFT_RECORDS=200`，被上限丢弃的计入
`drift_truncated`——被截断的清单不能读成清单很短）。**它不参与任何裁决，也不是开关。**
实测 211 份里 11 份有 drift（aria2 2 / bitcoin 9），共 368 行。

---

## 4. 适配器：一个文件一种容器

契约（`ReportAdapter`）只有两件事：

```python
class ReportAdapter(ABC):
    name: str; format: ReportFormat; analyzer: AnalyzerKind; suffixes: tuple[str, ...]
    def sniff(self, path, head) -> float:   # [0,1]，默认只看后缀
    def load(self, path, text) -> ReportDocument:  # 整个文档 → 全部 finding
```

- **注册**：`@register` 装饰器，`adapters/__init__.py` 里 import 即注册。
- **探测**：`sniff` 看**内容**（前 `SNIFF_BYTES` 字节），后缀只作辅助——因为 `.json` 同时是
  Infer、SARIF、Cppcheck 的容器，而 SARIF 可能**没有后缀**。
- **歧义是响亮的**：最高分并列 ⇒ 抛 `UnsupportedReportFormat`，消息里点名两个适配器并给出
  `--report-format` 的修法；只有严格更高分才胜出（弱一点的竞争者不阻断）。
- **失败是响亮的**：解析不了抛 `ReportParseError`，缺 finding 位置抛 `MissingReportMetadata`
  （位置是每个阶段的键，宁可拒绝也不按 line 0 分析）。

### 现状：四种分析器，两个适配器

| 分析器 | 原生格式 | 现在走的路径 |
|---|---|---|
| Clang Static Analyzer | HTML | `csa_html` |
| CodeQL | SARIF（原生） | `sarif` |
| Infer | `report.json` / `--sarif` | **`sarif`**（`--sarif` 输出） |
| Cppcheck | XML / `--output-format=sarif` | **`sarif`**（SARIF 输出） |

Infer 与 Cppcheck 的原生格式**没有**适配器，是**有意**的：两者都能产出 SARIF，而多一个只翻译
容器的适配器就多一处会漂移的解析。真要接原生格式时，判据是"SARIF 表达不了什么"——目前已知的
只有 Cppcheck XML 的 `<location file0= line0=>` 嵌套调用链和 Infer 的 `bug_trace` 级别标注，
都是**已有 SARIF 适配器不产出**的信息；到那时再加文件，不要把它们塞进 SARIF 适配器里。

---

## 5. 寻址：`ReportRef` = `<path>::<finding>`

一份 CSA 报告 = 一个文件 = 一个 finding，所以"哪份报告"和"哪个 finding"长期是同一个问题。
一份 SARIF 让它们分开，而**流水线的工作单元是 finding**：一个 finding 一份结果 JSON、一个 POC、
一个输出文件名。

```python
ReportRef.parse("/x/faiss.sarif::cpp/leap-year@AlignedTable.h:66")
# ReportRef(path=/x/faiss.sarif, selector="cpp/leap-year@AlignedTable.h:66")
```

`ReportRef` 实现了 `os.PathLike`（`__fspath__`/`name`/`stem`/`suffix`/`parent`/`exists`），
所以 CLI 与 `tools/` 里那几十处 `Path(report_path)`、`.stem`、`.name` **一行都不用改**。
`key` 是输出文件名的来源：无 selector 时**就是原来的文件名**（既有 CSA 结果 JSON 逐字节不变），
有 selector 时是 `<stem>__<净化后的 selector>`（有长度上限，且对所有语料 id 都是文件系统安全的）。

- `frontend.expand(refs)`：一个多 finding 文档 → 每个 finding 一个 ref；单 finding 文档 → 它自己；
  **读不了的文档原样透传**（让流水线给出带真实原因的 ERROR 行，而不是悄悄少跑一批）。
- `frontend.iter_findings(path)`：`(finding_id, str)`，`--list-findings` 用它。
- `frontend.load(ref)`：不带 selector 去读多 finding 文档会**报错**（不是随便挑一个）。
- `LoadedReport.summary()`：可序列化的摘要（格式/分析器/已解析文件数/弱匹配数/未解析数），
  阶段 1 打印的就是它。

---

## 6. 兼容层：`report_bridge`

流水线是 26k 行、按 `ParsedReport` 与那串字典写的。把它们改成读规范模型是**正确的终态**，
也是一次不可能一次落地的改动。所以流水线保留自己的接口，`fp_analysis/report_bridge.py`
就是那条缝——它把规范 finding 渲染回各阶段期待的那串字典。两条后果必须说清：

- **CSA 进、逐字节原样出**：每个 legacy 字段都从类型化步骤重建，其中三处（循环/三元的
  `branch_condition`）取自步骤的 `raw`，也就是**旧解析器产出的那个值**。`tests/test_report_frontend/test_csa_fidelity.py`
  用 `git show HEAD:` 抓下来的旧解析器输出做了黄金对照（25 行事件 + metadata + path id + 61 段
  source snippet），另有对 `*/reports/*/report-*.html`（211 份）逐份的语料不变量检查。
- **非 CSA 进、什么也不编**：SARIF finding 没有 `error start` 步骤、没有 `Calling 'X'` 措辞、
  没有分支方向，渲染出的字典就**不带**这些字段；找它们的阶段什么也找不到，于是停在保守态。
  为了让 CodeQL 输入"看起来像" CSA 路径而补值，正是**假 TP 被认证**的途径。

桥还负责一件事：**报告本身的步骤必须能被阶段 1 认出来**。`cfg_feasibility.build_bug_report_path`
历史上只认 `path_number == 9999`，而 SARIF 渲染不带这个值，于是 CodeQL finding 被切成 **0 段**。
现在两边都表达同一事实：桥对 `StepKind.REPORT` 写 `9999`，阶段同时接受 `event_kind == "report"`。
对 CSA 两者恒一致，所以第二个判据只可能确认第一个。

---

## 7. 谁在用

### 7.1 脚本 / 流水线入口（阶段 1）

`run_path_selection.py` 是运行器（argparse / `.env` / `main()` / 兼容再导出）；前端是在
`pipeline.stages.parse_report` 里被调用的，`ReportRef` 的展开在 `run_path_selection.main` 的
`_collect_reports`（`pipeline.batch`）一侧。

```bash
python3.10 run_path_selection.py --report <report.html | report.sarif[::<finding>]>
python3.10 run_path_selection.py --report x.sarif --report-format sarif
python3.10 run_path_selection.py --list-formats
python3.10 run_path_selection.py --list-findings --report x.sarif
```

- `--report-format` 钉住适配器（绕过探测歧义）；`--list-formats` 打印适配器表；
  `--list-findings` 打印一份文档里的 finding id / rule / 严重度 / 位置。这两个开关在
  **`.env`/API key 检查之前**就短路退出，所以它们永远不需要 key。
- 结果 JSON 与批处理汇总用 `report` / `report_file` / `finding` 三个字段：无 selector 时 `report`
  **仍是原来的文件名**（既有产物不变），有 selector 时用 `key`（62 个 finding 不会互相覆盖）。
- SARIF 没有函数名，于是 `_infer_entry_function(sdg, parsed)` 从 PDG 反推：取**同文件、行区间
  最紧包含 bug 行**的那个函数，写回 `parsed_report.metadata.function_name`（caller_map 遍历与
  CFG cache 的兜底都要用）。报告自己说了函数名时**绝不**二次猜测。
- 源根推断（`_infer_source_root`）的候选：报告上溯三级（`<project>/reports/{TP,FP}/x` 的布局）、
  `--projects-dir/<从文件名猜的项目名>`（`report-foo-1`→`foo`；`faiss.sarif`→`faiss`）、
  `--projects-dir/<stem>`、`--projects-dir/<stem 首个点之前>`、文档 `source_root_hint` 的
  basename/stem 尾部。每个候选都要通过 `_looks_like_project_root`；候选去重后按序试。
  **全都不通过就报错**，并把试过的候选逐条列进消息——「绝不替换成别的 project」这条不变量，
  对"推断源根"这一层同样成立。

### 7.2 `tools/build_project_deps.py`（PDG 构建的种子）

种子集决定**哪些 TU 被建成 PDG**，所以这里解析错一次，产物就是"缺了那份报告自己的函数"的图，
而下游**分不出**它和"分析器本来就没有这个函数"的区别。改动有三点：

- 解析走前端，因此 CSA 目录与 SARIF 目录用同一条路；`expand` 让一份多 finding 文档贡献多个种子，
  日志与 gap 报告用 `report.name::finding` 命名（无 selector 时就是原来的文件名）。
- **解析阶梯委托给前端**（`resolve_reported`），工具不再自带一套。曾经的后果很具体：工具自己那套
  只有"绝对路径 / 构建前缀 / basename"三级，**没有 relative 一级**，而 SARIF/Cppcheck 的 `uri`
  恰恰是相对路径——于是 CodeQL 语料的文件只能靠文件名猜。构建根前缀表也只剩前端一份
  （两份表会漂移，就像 `PROJECT_SPECS` 与 `cfg_parser._CFG_PROJECT_CONFIGS` 一直被 `--check-config` 抓到）。
- **弱匹配分级**：
  - **bug 文件**按名字猜中（`is_ambiguous`）⇒ 记为 GAP，消息里列出**全部**同名文件；
    这是报告自己的起点，猜错等于建错 TU；
  - **事件文件**歧义 ⇒ 保留。它只是**加宽**构建，而分析的起点是 host TU。
- `--on-demand <dir>` 同时 glob `report-*.html` 与 `*.sarif`（两者都没有时退回全部 `*.html`）。

### 7.3 代码模型的读取消费者：报告是索引，模型是读者

规则一句话：**凡是交给 LLM 或闸门的项目代码，都必须由代码模型给出**。报告模型只回答"哪个文件、
哪几行"，代码模型回答"那是什么代码"——所以解析阶梯、读文件、定位函数、取窗口都只有一份实现，
调用方按用途取用（把结果当权威的只收强策略，见 §3）。

| 消费者 | 位置 | 从模型取什么 |
|---|---|---|
| 分段 / CFG 归因 | `cfg_feasibility._resolve_local_path`、`_resolve_source_file_for_cfg` | 路径解析 |
| 6a 自相矛盾门 | `report_consistency._read_source` | claim 所在文件的行走 `model.lines` |
| 6b 入口状态门 | `entry_state.detect_fabricated_entry_state` | 用 `code_model_of(parsed)` 解析 bug 文件、`model.lines` 读它；弱命中返回 `None`（"查不了"，不是"没有"） |
| 路径渲染 | `cfg_models.BugReportPath.source_between` | `code_model.lines` |
| constraint-completion prompt | `path_analyzer._build_event_source_index` | 事件源码索引 |
| POC 生成 / 审计 prompt | `source_context.SourceContextExtractor` | 主文件窗口、companion 窗口、被调函数体（`function_files` 先问 PDG 索引，再问树索引；不再 `rglob` 猜、不再拼 `-inl.h`） |
| 模拟验证的步骤清单 | `simulation_verifier._csa_step_rows` | 按 **(file, line)** 查——此前只按行号，文件 B 的步会看到文件 A 的同行 |
| 兼容层片段 | `report_bridge.source_snippets` | `loaded.code.snippet()`（embedded 优先，`raw` 仅兜底） |
| 脚本 / 流水线入口 | `run_path_selection.main`、`pipeline.runner.process_report`、`pipeline.stages.parse_report`、`pipeline.project_paths._resolve_paths` | `ReportFrontend.load`/`expand`（一个 finding 一个工作单元）、`ReportRef` 的 `os.PathLike` 表面、`LoadedReport.code`（结果 JSON 的 `code_model` 块由它出） |

**三处声明式例外**（不是漏改，理由在此，避免下次被"顺手统一"）：

- `entry_state.scan_call_sites` —— 全树搜"这个函数被谁调用"，它要找的正是**报告没有提到的文件**；
  结果是**旁证**（进 reason，永远不能单独支撑裁决）。走模型就等于让模型常驻整棵树。
- `cfg_feasibility._build_class_to_file_map` —— 只做 `is_file()` 探测、**不读代码**；模型的按名工具
  是**全树索引**，放宽探测范围会把 `<Name>.cc` 在别处的类名也纳入，那是**改变分段**而不是改变读取。
  它的搜索目录（bug 文件所在目录 / `src/` / 根）原样保留。
- `constraint_miner` —— 它 `rglob` 全树建**跨报告复用**的约束知识库，报告索引根本不在回路里；
  把语料级产物绑到单份报告的模型上是错的。

**内存**：materialize 的预读量就是"报告相关代码"的量级。PDG 在场时实测一份报告 **3 个文件 /
22.8 KB**；最坏一次（无 PDG、被调函数查找失败的整树扫描）1445 文件 / 24 MB 文本，暂态、随报告
结束释放，而 `pdg_<project>.json` 本身常驻 38–54 MB。结果 JSON 的 `code_model.code_set` /
`code_model.lazy_reads` 是这两个数的出口：`code_set` 说预读装下了什么，`lazy_reads` 说之后
又补读了什么——后者不为空就说明这一轮的内存不是由"报告提到的文件"封顶的。

---

## 8. 不变量（防回归）

1. **一种格式一份适配器，绝不"翻译成另一种格式"**。适配器只做 `容器 → 规范模型` 的归一化。
2. **模型里没有的东西，下游拿不到**；缺失信号必须产生**保守默认**，不许产生一个自信的结论。
3. **解析阶梯只有一套**（`code_model.SourceCodeModel`），所有消费者共用；构建侧尤其不许自带第二套。
   现有消费者：`cfg_feasibility._resolve_local_path`、`cfg_cache_set`、`build_project_deps`、
   `run_path_selection._resolve_source_file_for_cfg`、`report_consistency._read_source`、
   `source_context._resolve_to_local`、`entry_state._resolve_local`、`cfg_models.source_between`、
   `path_analyzer._build_event_source_index`、`simulation_verifier._csa_step_rows`。
   **新增路径解析前先找这里**——第 256 行的
   教训是这套阶梯曾经"看起来只有一套"，而实际上 `cfg_feasibility` 里还留着一份只会剥前缀的
   旧实现：CSA 的绝对路径它照样能过，所以 211 份 CSA 语料全绿，只有真 CodeQL 报告进来才暴露
   （`src/fmt.cc` 解析为 `None` → 事件掉进按行号匹配 → 归因到无关文件）。
   **同理，"读代码"也只有一套**（§7.3）：除三处声明式例外（`scan_call_sites`、
   `_build_class_to_file_map`、`constraint_miner`），管线里没有第二处 `read_text` 读项目源码。
4. **`basename` 是弱策略**，`STRONG_STRATEGIES` 不含它；**管线里按用途取用**：
   `_resolve_local_path` 这类"把结果当权威"的消费者只接受强策略，弱命中返回 `None`（6b 的
   `entry_state._resolve_local`、B8 的 `source_context._resolve_to_local` 同一条规则，写法一致——
   同一问题的两个接缝不许对"解析成功"有不同定义）。兜底比错觉好。歧义时用 `rivals`
   （不含选中项），报"全部同名文件"时用 `candidates`。
5. **CSA 输出逐字节不变**：改动兼容层后必须跑 `test_csa_fidelity`（黄金对照 + 211 份语料不变量）。
6. **多 finding 文档必须逐 finding 寻址**：`expand` + `ReportRef.key` 是唯一入口；输出文件名、
   结果 JSON 身份、批处理汇总都从这里取，不许回到"报告文件 = 工作单元"。
7. **`REPORT` 步骤必须在两条表示里都成立**：桥写 `path_number=9999`，阶段接受
   `event_kind == "report"`。任一被删，非 CSA 报告会被切成 0 段（静默，不是报错）。
8. **两个开关不加第三条**：`--report-format` 是覆盖探测歧义的逃生口，`--list-formats` /
   `--list-findings` 是自查口；不要把适配器选择做成常规模式开关（与「无模式开关」一致）。
9. **失败要响**：探测歧义、解析失败、缺位置、报告找不到——各自抛 `errors.py` 里的异常，
   消息里带修法。
10. **"有没有路径"是模型里的事实，不是数出来的**：`BugFinding.path_evidence` 由适配器给出，
    默认 `LOCATION_ONLY`；消费者不许用 `len(steps)` 反推。下游据此分流（阶段 2 对
    `location` 只取报告位置所在函数；阶段 10 对 `location` 不作 POC 验证、判 UNKNOWN，
    理由见 `docs/pipeline-stages.md` §8 不变量 17），矛盾时打 warning 而不是静默二选一。

---

## 9. 验证基线

新测试包 `tests/test_report_frontend/`（441 个用例），与既有用例合计 **1109 passed**：

| 文件 | 覆盖 |
|---|---|
| `test_models.py` | 枚举归一、`StepKind.is_boundary`、`TraceStep` 往返（含 `raw`）、`BugFinding` 访问器、finding selector 的各种写法 |
| `test_registry_and_detection.py` | 注册表、内容探测（无后缀 SARIF、Infer SARIF）、未知格式报错、**同分歧义必须响亮**、弱竞争者不阻断、`--report-format` 覆盖 |
| `test_sarif_adapter.py` | 文档/finding/严重度/类别/指纹、重复位置 id 加后缀、**不发明 CONTROL/ASSUME**、flow 顺序、`[Return]` → `RETURN`、最长 flow 胜出、坏 JSON、suppressions 保留 |
| `test_csa_fidelity.py` | `git show HEAD:` 抓下的黄金对照 + **211 份**真实 CSA 报告的语料不变量 |
| `test_code_model.py` | 阶梯每一级、basename 歧义与 `rivals`、`STRONG_STRATEGIES`、读代码、`references` 角色 |
| `test_frontend_and_refs.py` | `ReportRef` 解析/`os.PathLike`/`key` 净化与限长、`expand`、加载、未解析告警、`iter_findings`、钉格式覆盖探测 |
| `test_bridge.py` | 两种格式都产出的 legacy 键集、非 CSA 不臆造、`path_length` 来源、snippet 顺序、`path_id` 唯一、`to_bug_path` 末尾必是 REPORT 且不重复、**阶段 2 真能吃下 SARIF** |
| `test_cli_integration.py` | 工作单元身份（无 selector 时文件名不变）、输出命名、`_collect_reports` 展开、源根推断候选与 fail-fast、`_infer_entry_function`、PDG 版本闸门、`--list-formats`/`--list-findings` |
| `test_build_tool_seeds.py` | 种子（bug 文件/事件文件/callee/对齐）、多 finding 展开、坏报告成 GAP 而非崩溃、弱匹配分级、解析策略、host TU |

**代码/语料实测**（本次重构后）：

```bash
python3.10 -m pytest tests/ -q
python3.10 tools/replay_steps1to6.py <回归集>          # 45/45 段解析, 10136 分支, 1 unresolved
python3.10 tools/abc_compare.py      <回归集>          # clone_index 仍 2 段 / 4 分支 / bound 16
```

**"读码通道"收敛后的两条实测基线**（2026-09-29，`source_context`/`entry_state` 改走代码模型）：

| 通道 | 改动前 → 改动后（同一批报告） |
|---|---|
| POC prompt 的三块文本（主文件窗口 / 调用链 / 被调函数体，7 份） | **6 份逐字节相同**，1 份不同且是**修正**：`fourcc_inv_printable-5-1` 原先拿到的是**错函数**的 35 行（把 `read_VectorTransform` 里的调用点当函数体），现在由 PDG 索引定到 `faiss/impl/io.cpp` 的真定义 |
| 6b 入口状态命中集合（protobuf/faiss/aria2/folly 共 53 份） | **逐字节相同**（仍 2 份：`numbers.cc-SimpleAtob-{11,5}-1`，解析出的文件路径也相同） |
| 树内打开的文件数（模型外的 `read_text` 计入同一口径） | 69→4、85→6、4543→1445、2025→1495、54→2、62→4、6→2 |


- **种子集回归**：把 HEAD 的工具解析器与新的前端阶梯对同一批报告各跑一遍，比较
  `bug_file / local_file / event_files / callees / 对齐比 / problems / host TU` ——
  faiss(13) / folly(8) / aria2(11) / protobuf(21) 共 53 份报告**全部 0 处变化，TU 集完全相同**。
- **A/B/C 与 bound**（`abc_compare`，A/B/C 关 vs 开）：`clone_index` 保持 2 段 / 4 分支 / bound 16；
  `484-2` 保持 45 段 / 10136 分支、bound 不变（A 在 faiss 上仍是 inert，原因见
  `pipeline-stages.md` §5 的 block-id 对齐）。
