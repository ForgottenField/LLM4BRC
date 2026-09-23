# C++ 项目依赖图构建指南（PDG + CFG cache）—— 以 faiss 为例的统一方法

> 本文档记录「为一个 CSA 目标 C++ 项目构建本流水线所需依赖图」的**完整、可复用流程**：
> 如何生成 `compile_commands.json`、构建 `pdg_<project>.json`、离线预热 `cfg_cache/<project>/`，并注册工程配置。
> 以本次 **faiss** 的实际构建为实例，最后给出**统一化（多项目一键式）**的分析与推荐方案。
>
> 项目：`bug_report_classification`
> 运行环境：Linux；`python3.10`（pytest/PYTHONPATH 均以 3.10 为准）；`clang++-14` / `llvm-config-14`；系统 cmake 3.16（faiss 因此无法 cmake 生成编译数据库，需手写）。

---

## 0. 为什么每个项目需要「三件套」

`run_path_selection.py` 分析一份 CSA 报告时，需要目标项目的三种结构数据。缺一不可：

| 产物 | 位置 | 由谁消费 | 内容 |
|---|---|---|---|
| **`pdg_<project>.json`** | 仓库根 | `load_sdg`（`run_path_selection.py:215`）→ 切片 / 分支提取 / 函数归因 | 每个函数的程序依赖图（节点带 `source_line`/`block_id`/`kind`，边有 `control_dep_edges`/`def_use_edges`） |
| **`cfg_cache/<project>/*.json`** | `cfg_cache/<project>/` | `CFGCacheSet`（`fp_analysis/cfg_cache_set.py`）→ 分段后按路径文件**按需加载并合并** | 每个**编译单元**的全 TU CFG（按 `md5(绝对源路径)` 命名） |
| **工程配置** | `cfg_parser.py` 的 `_CFG_PROJECT_CONFIGS` | 运行时 `build_cfg_command` | 每个项目的 include / define / force_include |

顶层 `run_path_selection.py` **不会**在运行时自动构建 PDG 或 CFG cache：
- PDG 必须预先存在（`load_sdg` 直接读文件）；
- CFG cache 只做**查找**（按 md5 glob），缺失就 `WARNING: no CFG cache for <file>` —— 不会现场 dump。

因此**必须离线预热**。CFG 的运行时自动 dump 逻辑在 `cfg_feasibility.py` 里存在，但顶层入口不走它。

> CFG cache 是**按编译单元**存的，而 bug path 常跨文件：顶层加载策略是「先 bug 宿主
> 文件的 cache，分段后按每个 segment 的 `function_file` 逐个 `ensure()` 合并」（见
> `fp_analysis/cfg_cache_set.py`）。所以一个项目要覆盖其 bug 报告，需要把报告里出现的
> **所有** `.c/.cc/.cpp` 编译单元都预热，只建 bug 宿主那一个会让跨文件段退化成 linear
> （这正是 `report-clone_index.cpp-operator=-2-1` 路径空间被算成 1 的原因之一）。
> 合并时 `raw_dump` 会被丢弃（占 cache 体积 ~95%，全仓库无消费者）。

> 本 repo 里 folly / aria2 / faiss 的三种产物即按此模式预先构建完成，可直接对照。
> 唯一与流程无关的既有弱点提示：即便依赖图构建正确，工具对「CSA 已全解析的密集微 segment」可能找不到可反证分支而判 TP（见 memory `folly-v2-tp-bias` / `faiss-dependency-build`）——那是**评估性/判定性**问题，与依赖图是否建对无关。

---

## 1. 整体流程一览

```
[项目源码树 /csa_reports/project/<proj>]
        │  (1)
        ▼
[compile_commands.json]  ──►  (2) pdg_builder  ──►  pdg_<proj>.json   （放仓库根）
        │
        │  (1')
        ▼
[列 bug 宿主 .cpp]        ──►  (3) cfg_feasibility 离线 dump+parse+save  ──►  cfg_cache/<proj>/*.json
        │
        │  (4)
        ▼
[cfg_parser._CFG_PROJECT_CONFIGS 增 "proj" 配置]  ──►  (5) 冒烟验证
```

- **(1)/(1')** 都吃「源文件 + 每条翻译单元的编译参数」——所以**最上游的单一输入是 compile_commands.json**。
- **(2)** 由 C++ 工具 `pdg_builder` 完成，命令行固定 `--compile-db <db> --output <out>`。
- **(3)** 复用 `cfg_feasibility.build_cfg_command/run_cfg_dump/parse_cfg_text/_save_cfg_cache`，逐个源文件跑一次 `clang++-14 ... debug.DumpCFG`。

核心洞察：**只要 (a) 拿到一套正确的 compile_commands.json，(b) 能在 `clang++-14 -std=c++17` 下语法通过，(c) 登记工程配置——三件套即可对任意 C++ 项目离线生成**，且 (2)/(3) 与具体项目无关、只需参数。

---

## 2. 步骤一：生成 `compile_commands.json`（最易踩坑的一步）

PDG 工具 `pdg_builder` 走 `tooling::JSONCompilationDatabase`，即把每条编译指令交给它去喂给 clang AST。它只认 **clang 能真正编译通过的 TU**；任何编不过的 TU 要么剔除、要么单独修。

### 2.1 何时能 / 何时不能走 cmake

- **能**：项目自带的 cmake ≥ 3.17（faiss 需要）且能正常 configure/bear 捕获 → 直接 `bear -- make` 或 cmake `CMAKE_EXPORT_COMPILE_COMMANDS=ON`。
- **不能**（本次 faiss 情形）：系统 cmake 3.16 < 3.17 → **手写**编译数据库，仅对**你关心的范围**列条目。

### 2.2 手写编译数据库：范围选取

先确认目标范围。faiss 选了**全核心**（非 gpu / 非 python / 非测试的 `faiss/**/*.cpp`）+ `tests/test_merge.cpp`（一份报告的宿主）。范围取舍：

- **核心 `.cpp`** 通常够（bug 报告集中在此）。
- **gpu / 绑定 / 可选依赖 / 巨型测试头** 往往含环境专属头或编译冲突，默认排除，命中报告时再单独补。
- 判据简单：**该文件是否是你那批 bug 报告的宿主或关键被调函数所在**。

### 2.3 手写条目格式与「先验证后写入」

每条 entry：
```json
{ "file": "/abs/path/f.cpp", "command": "clang++-14 -std=c++17 <flags> -c /abs/path/f.cpp -o /dev/null" }
```
写库前**先逐条 `-fsyntax-only` 验证**，只把通过的写入库；未过的打日志（faiss：76/77 通过，仅 `impl/ScalarQuantizer.cpp` 失败）。

**faiss 的 flags 是如何定出来的（通用排查套路）**：
1. **`#include "faiss/..."`（项目内相对包含）→ `-I<project_root>`**。
2. **`omp.h` 找不到** → `-I/usr/lib/gcc/x86_64-linux-gnu/10/include`（gcc 自带头，含 omp.h）。*注：曾试 clang-16 的同名目录反而破坏 cstdint/stdint 解析，回退 gcc-10。*
3. **`FINTEGER` 未定义** → 参考同库其它文件里的 `#define FINTEGER int`，统一加 `-DFINTEGER=int`。
4. **个别 SSE/内建冲突**（gcc 头遮蔽 clang 内建）→ 单文件剔除并记录，只削弱该文件切片，不阻塞整体。

> 提示：这三个 flag 里 2、3 本质是「被系统的环境细节」；真正的项目专属只有 1。这就是下面统一化方案里配置表想固化的东西。

---

## 3. 步骤二：构建 `pdg_<proj>.json`

```bash
cd /home/yanghengqin/bug_report_classification
src/llm_client/csa_analysis/pdg/build/pdg_builder \
  --compile-db /tmp/faiss_cc/compile_commands.json \
  --output pdg_faiss.json
```

**验证三件事**：
1. 文件生成、体积合理（对照：aria2 46MB/24019 fn、folly 21MB/10840 fn、faiss 53MB/13648 fn、protobuf 38MB/12796 fn）。
2. 可用 `run_path_selection.load_sdg("pdg_faiss.json")` 载入，函数数非 0。
3. **该批报告的 bug 宿主函数都在册**（判断有没有走 permissive fallback）：
   ```python
   import json; d=json.load(open("pdg_faiss.json"))
   names={f["function_name"] for f in d["functions"]}
   assert "faiss::read_index" in names
   ```
   注意 PDG 函数名是 **fully-qualified**（`faiss::read_index`），而 CFG cache 键是**带签名的原始串**（`faiss::Index *read_index(faiss::IOReader *f,...)`）——两者命名不同属正常；运行时靠 `sdg.get_pdg` 的精确末段 `::` 匹配对齐（不会把 `read_index` 误配到 `read_index_binary`）。

构建工具本身是 C++，需先编译（通常已就绪）：
```bash
bash src/llm_client/csa_analysis/pdg/build.sh   # 产出 build/pdg_builder（唯一二进制名，勿另起名）
```

### 3.1 产物版本必须与代码一致（陈旧产物的教训）

`PDGBuilder.cpp` 在输出里写了 `Metadata["version"]`（当前 `1.3`），Python 侧
`pdg_models.PDG_ARTIFACT_VERSION` 必须与它相同。**陈旧产物解析不会报错，只有这个声明能说明**——
`pdg_protobuf.json` 曾以 v1.0 放了数周（二进制是旧的 `pdg_builder_local`，8-21 版），对比同一份
compile db 重建的 v1.2：控制依赖边 23,746 → 33,729（+42%），且分支谓词从宏记账节点改成真正的谓词
（`v.size()` → `!(!((v.size()) <= (kStringPrintfVectorMaxArgs)))`），整个 21 份报告的那轮评估因此作废。

版本历史（每次形状变化都必须提版本，Python 常量同步）：

| 版本 | 内容 |
|---|---|
| 1.0 → 1.1 | 宏展开分支标注（`is_macro` / `macro_name` / `macro_def_*`） |
| 1.1 → 1.2 | postdominator 自身成员修复（链式控制依赖）+ 分支谓词取 terminator 条件 |
| 1.2 → 1.3 | **表达式不再截断**（原 `> 120` 即砍成 117 + `"..."`，3 处：recovery-expr / 普通语句 / 不在 CFG 的节点） |

**为什么去掉截断**：截断点正好落在机器生成的长名字上——模板实例化与宏展开，于是**互不相同的条件
变成逐字节相同**：五条不同的 `EXPECT_EQ` 记下同一段 `AssertHelper(...)` 前缀，阶段 6 据此报出
一个不存在的矛盾。去除后的实测（protobuf，21 TU）：函数 12796 / 节点 77702 / CD 边 33729 /
DU 边 27118 **与 v1.2 逐项相同**（截断纯文本层面，不参与切片与冲突判定），体积 38.08 → 38.78 MB
（+1.8%），最长表达式 2125 字符、p99 337、平均只超 111 字符；faiss 同样形状不变（8906 处截断 → 0，
最长 1636）。**需要短文本的消费者（prompt、结果 JSON）应在渲染时截断，而不是让产物存残缺文本。**

三道闸门（都在工具/流水线里，不是靠人记得）：

| 位置 | 行为 |
|---|---|
| `tools/build_project_deps.py` 构建前 | `check_builder`：二进制缺失、源码声明版本 ≠ `PDG_ARTIFACT_VERSION` → **拒绝构建**；二进制早于源码 → WARNING |
| 同上，构建后 | `verify_pdg_version`：产物声明版本不符 → 报错并**删除该产物**（留着就是「一份错文件，日后被复用」） |
| `run_path_selection._resolve_paths` | 分析任何报告前校验 `pdg_<proj>.json` 的声明版本，不符 → **直接报错退出**（不分析、不降级） |

随时可查全仓库产物状态（无需指定项目）：

```bash
python3 tools/build_project_deps.py --check-config   # 打印每个 pdg_*.json 的声明版本 + OK/STALE
```

**一个项目只留一份产物**：不要把旧产物改名成 `.bak` 留在仓库根——`--check-config` 会把它当
STALE 报出来，而任何按名字取用 `pdg_<proj>.json` 的脚本都不会看 `.bak`。要换版本就地重建、旧文件删除。

---

## 4. 步骤三：离线预热 `cfg_cache/<proj>/`

CFG cache **每文件 = 一个源文件的整 TU CFG**，文件名为 `cfg_cache/<proj>/<md5(绝对源路径)>.json`（`cfg_feasibility._cfg_cache_path`）。顶层只按报告解析出的 bug 宿主源文件去查找，所以**至少要给「报告实际解析到的源文件」建缓存**。

### 4.1 确定要建哪些源文件

跑一次报告，从日志/解析结果看 bug 宿主文件；常见有歧义点：
- 报告前缀是 `.cpp`（`report-index_read.cpp-...`），但事件行可能落在**被调函数所在的另一个 `.cpp`**（faiss 的 `fourcc`→`impl/io.cpp`、`scan_codes`→`IndexIVF.cpp`）。
- 极少数解析到**头文件**（`AlignedTable.h`/`Heap.h` 这类模板/内联宿主）——此时应按 folly 经验：**从实例化的 `.cpp` 建 CFG，再存到该头文件解析出的 hash 名下**。可先跳过头部报告，仅保证 `.cpp` 宿主覆盖（faiss 的 8 个报告即靠 10 个 `.cpp` 覆盖）。

稳妥做法：**把「报告前缀命名的 `.cpp`」与「解析结果指出的 `.cpp`」做并集**，全建整 TU 缓存。faiss 最终集合：
`IVFlib.cpp, IndexIVF.cpp, IndexPQ.cpp, IndexRefine.cpp, IndexScalarQuantizer.cpp, IndexFastScan.cpp, clone_index.cpp, impl/index_read.cpp, impl/io.cpp, tests/test_merge.cpp`（10 个）。

### 4.2 复用 cfg_feasibility 离线 dump（faiss 实际用的代码）

```python
import sys; sys.path.insert(0, "src")
from llm_client.fp_analysis import cfg_feasibility as CF

SRC = "/home/yanghengqin/csa_reports/project/faiss"
for rel in ["faiss/impl/index_read.cpp", "..."]:          # 见 4.1 的并集
    bf = f"{SRC}/{rel}"
    cache = CF._cfg_cache_path(bf, "faiss")
    if cache.exists():
        print("cached:", rel); continue
    cmd  = CF.build_cfg_command(bf, "faiss", SRC)          # 用已登记的配置构造 clang 命令
    raw  = CF.run_cfg_dump(cmd)                            # clang++-14 ... debug.DumpCFG
    fns  = CF.parse_cfg_text(raw)
    CF._save_cfg_cache(cache, fns)
    print("OK", rel, len(fns))
```

> 前置：`cfg_parser._CFG_PROJECT_CONFIGS` 里必须已登记 `"faiss"`（见步骤四），否则 `build_cfg_command` 补不上 `-I<root>`/omp/`FINTEGER`，dump 会失败或残缺。

### 4.3 建完后自查

确认每个缓存文件里的**目标函数真的带分支块**（而不只是空壳）：载入后 `CFGFunction.blocks` 数量应与源码复杂度匹配（faiss `read_index(IOReader)` → 1308 blocks / 532 个 >1 后继块）。若某函数找不到，多半是工程配置缺 include/define 导致解析降级。

### 4.4 缓存体积：`raw_dump` 不再落盘（2026-09-22 起）

`parse_cfg_text` 会把**整个 TU 的 dump 原文**塞进**每一个** `CFGFunction.raw_dump`
（原意是调试可读），于是缓存体积 = `函数数 × dump 大小`，而不是 `dump 大小`：

| TU | 函数数 | dump 大小 | 旧缓存 | 新缓存 |
|---|---|---|---|---|
| faiss `clone_index.cpp` | 少 | ~3 MB | 14.1 MB | 0.75 MB |
| protobuf `text_format.cc` | 3290 | 3.3 MB | **10.9 GB** | 7.1 MB |
| protobuf `util/message_differencer.cc` | 3243 | 3.3 MB | **10.7 GB** | 7.1 MB |

`_load_cfg_cache(..., drop_raw_dump=True)` **早就在读入时丢掉这个字段**，全仓库无消费者，
所以写入它只是白占磁盘 + 让每次运行多一次 10 GB 的 `json.loads`。现在 `_save_cfg_cache`
默认 `drop_raw_dump=True`，不再写这个字段（读入侧兼容新旧两种缓存，旧缓存仍可读）。
protobuf 全套缓存因此从 **30.8 GB 降到 35 MB**；faiss/folly 的既有缓存未重建（多占的只是磁盘，
读入行为不变），下次按需重建时自然收敛。

---

## 5. 步骤四：登记工程配置（两处都要加！）

运行时 CFG dump 与验证阶段的 POC 编译各自维护一份同名配置，**漏一处就会有一侧失败**：

1. `src/llm_client/fp_analysis/cfg_parser.py` → `_CFG_PROJECT_CONFIGS`：
   ```python
   "faiss": {
       "include_paths": [".", "/usr/lib/gcc/x86_64-linux-gnu/10/include"],  # "." 相对 project_root 解析到 -I<root>
       "defines": ["FINTEGER=int"],
       "force_includes": [],
   },
   ```
2. `src/llm_client/fp_analysis/cfg_parser.py` → `_CFG_PROJECT_CONFIGS`：同样加 faiss 键。

`build_cfg_command` 对 `include_paths` 会先尝试 `Path(source_root)/rel_inc` 是否存在再拼 `-I`，所以应指向能解析 `#include "faiss/..."` 的根目录。

---

## 6. 步骤五：冒烟验证

快跑 2–3 份报告（一份 TP 目录 + 一到两份 FP 目录），核对：

```bash
set -a; . ./.env; set +a     # DEEPSEEK_API_KEY
python3.10 run_path_selection.py --project /home/yanghengqin/csa_reports/project/faiss \
  --report .../reports/FP/report-index_read.cpp-read_index-484-1.html \
  --output output/faiss_smoke_readindex.json
```

看三处：
1. 无 `No CFG cache found` / `function not in CFG cache — skipping`（`FAISS_THROW_FMT` 这类错误内联缺失可忽略）。
2. 分段正常（`Segmented into N segment(s)`）。
3. 能产出 verdict，且 `cfg_cache/<proj>/` 不再增长（缓存命中）。

> **解读判定结果时要区分两类问题**：
> - 「依赖图没建对」→ 报错、跳过、函数缺失、缓存不增长。
> - 「建对了但判 TP / 0 分支」→ 是**判定层**的已知偏置（密集微 segment 无独立分支可反证），不是依赖图问题。见第 0 节末。

---

## 7. 统一化分析：能否一套流程覆盖多项目？

**结论：能，且应做成「一个通用脚本 + 一份每项目参数表」。** 理由：第 2–5 步里只有两类东西是项目相关的——

| 项目相关（需每项目配置） | 项目无关（可复用同一套代码） |
|---|---|
| ① 报告宿主的**源文件清单**（建哪些 CFG） | ② `pdg_builder` 调用（固定 CLI） |
| ③ **include / define / force_include**（进 compile_commands 与 `_CFG_PROJECT_CONFIGS`） | ④ `build_cfg_command/run_cfg_dump/parse_cfg_text/_save_cfg_cache` 全套 |
| ⑤ 手写 compile_commands 时**需要额外系统头/宏**（omp.h、FINTEGER 之类） | ⑥ `load_sdg` / md5 缓存命名 / 顶层查找逻辑 |
| ⑦ **源码根路径** `project_root` / `source_root` | ⑧ 判空/建目录/验证等胶水 |

### 7.1 推荐的统一形态：`config.py + 一个 build 脚本`（已实现 → 见 §9）

把项目相关收进**一张声明式配置表**（新增项目=加一行），项目无关的收敛成一个 `python` 脚本：

```python
# deps_config.py —— 每个目标项目一行
PROJECTS = {
    "faiss": {
        "root": "/home/yanghengqin/csa_reports/project/faiss",
        "compile_db": "/tmp/faiss_cc/compile_commands.json",   # 或留空→自动生成
        "cfgs": ["faiss/IVFlib.cpp", "faiss/IndexPQ.cpp", ...], # bug 宿主 .cpp 并集
        "cfg_parser": {   # 同时写入 cfg_parser._CFG_PROJECT_CONFIGS 的值
            "include_paths": [".", "/usr/lib/gcc/x86_64-linux-gnu/10/include"],
            "defines": ["FINTEGER=int"], "force_includes": [],
        },
        "cmake": {"min": "3.17", "flags": [...]},   # 能 cmake 就自动生成 compile db
    },
    "aria2":  { ... }, "folly": { ... },   # 可回填既有经验
}
```

统一脚本 `build_project_deps.py --project faiss [--compile-db PATH|--auto-cmake]` 执行：
1. （若 compile_commands 缺失）跑 `bear` 或 cmake 生成；否则读手写库。
2. 逐条 `-fsyntax-only` 验证→剔除失败的→写临时库。
3. `pdg_builder --compile-db ... --output pdg_<proj>.json`。
4. 对 `cfgs` 每个源文件跑 cfg dump 预热 `cfg_cache/<proj>/`。
5. 自动把配置 patch 进 `cfg_parser._CFG_PROJECT_CONFIGS`（或改成读外部 config 而非硬编码）。
6. 校验：PDG 函数数、缓存可载入、目标 bug 函数在册。

### 7.2 关键取舍 / 难点

- **compile_commands 生成策略**：优先项目自带 cmake≥3.17 → `CMAKE_EXPORT_COMPILE_COMMANDS=ON` 或 `bear`；否则**手写**。手写时「全核心 vs 最小集」要有开关（faiss 全核心 77 TU 已是上限，可放宽/收紧）。`ScalarQuantizer.cpp` 这类编不过的，用**排除名单**而不是硬改全局 flag——单文件修只能内联到该 TU。
- **工程配置落点**：目前硬编码在两个 python 模块里。更干净的统一做法是把 `_CFG_PROJECT_CONFIGS`/`_PROJECT_CONFIGS` 从「硬编码字典」改成「读 `deps_config.py`（或同名 json）」，这样新增项目零代码改动、无需 patch 源码。
- **CFG cache 覆盖范围**：建议脚本默认按「报告前缀 `.cpp` ∪ 解析结果 `.cpp`」自动收集；头部宿主仍单独提示。
- **隔离命名空间 / 工具链差异**：gpu/python/绑定目录、C++11 项目用 C++17 编译、个别 SSE 冲突——都是每项目排除名单的输入，属配置表数据，不属脚本逻辑。
- **可回填复用**：aria2（已工作，能剪出真 FP）与 folly/faiss（已知 v2-TP-bias）的差异正好可作为「工具对项目保真度敏感性」的对照样本，放进配置表注释里，提醒评估结果的解读。

### 7.3 收益

- 新项目从「手工 ~6 步、踩 cmake/omp/FINTEGER/SSE 一堆坑」降为「填一行配置 + 跑一个命令」。
- PDG 构建、CFG 预热、配置登记、校验四件事**幂等可重入**（compile db 落盘，续跑覆盖同输出）。
- 与现有 folly/aria2/faiss 三条经验并存，可反向回填为回归基线。

---

## 8. faiss 实际产物速查（本会话交付）

| 项 | 值 |
|---|---|
| `compile_commands.json` | `/tmp/faiss_cc/compile_commands.json`（76 核心 TU，不入库） |
| `pdg_faiss.json` | 仓库根；37.7MB；13648 函数 |
| `cfg_cache/faiss/` | 10 个整-TU 缓存（见 4.1 清单） |
| 工程配置 | `cfg_parser._CFG_PROJECT_CONFIGS` 已加 `"faiss"` |
| 冒烟 | 3 份报告端到端跑通；TP 目录 clone_index `operator=` 判 TP 对齐；两份 FP 目录判 TP（已知 v2-TP-bias，非构建问题） |
| 后续 | 已决定只保留依赖图、不跑 faiss 全量回归；随时可按 §6 命令补跑单份 |

---

## 9. 按需构建策略（on-demand）—— 只建报告用到的 TU

> 工具：`tools/build_project_deps.py`（本文档第 2–5 步的统一实现）。protobuf 是第一个按此策略构建的项目。

### 9.1 为什么需要按需

faiss（77 TU 全核心）之后再往后的目标（protobuf / grpc / duckdb）源码树大得多——protobuf 检出 270 MB，
abseil 子模块更大。而流水线**从不遍历整棵树**：它只看 CSA 报告走过的文件，以及这些文件里可能被展开的
被调函数（阶段 5 的 callee 展开）。所以「全量构建」的绝大部分算力花在永远不会被读到的函数上，
而且白建出来的函数还会稀释函数归因（同名/尾名同段匹配）。

按需构建把范围收到两个来源：

1. **种子（seed）**：报告自身给出的 TU —— bug 宿主文件 + 报告路径上出现的**每一个** event 文件。
   种子是**翻译单元**，不是函数：PDG 工具以 TU 为单位走 AST（它没有文件过滤开关，`--compile-db / --output / --verbose`
   之外没有参数），**范围完全由 compile db 的内容决定**。
2. **被调闭包（callee closure）**：报告事件描述里的 `Calling 'X'` / `Returning to 'X'` 提到的被调者，
   若其定义（ctags，非 grep）落在种子之外的 `.cc`，把那个 TU 补进来。

protobuf 实测：21 份报告 → **17 个种子 TU**，闭包再补 **4 个**（`absl/strings/cord.cc`、`absl/time/format.cc`、
`protobuf/message_lite.cc`、`absl/debugging/stacktrace.cc`），共 **21 TU**，而源码树里 protobuf 核心本身就有 76 个 TU。

### 9.2 三条不变量（比省算力更重要）

- **行号对齐是「验证」出来的，不是假设的。** 报告 HTML 内嵌了 bug 文件（及路径文件）的源码行，
  `ParsedReport.source_snippets` 可以当作**该版本源码的 oracle**：本地检出的第三三方/子模块若与报告生成时的
  版本不同，哪怕只差 2 行，后续所有 `(file, line)` → PDG 节点的配对都会错位（而流程**靠行号**配对）。
  因此建库前在 bug 行 ±8 行窗口内做一次偏移搜索：**非零偏移 = 失败**（不是「匹配」），
  命中率低于 `--min-line-match`（默认 90%）的种子被列为 GAP 并**拒绝**，绝不静默换成别的文件。
  - 两个易错点：① `source_snippets` 的键是**段落局部行号**，一份报告每个文件一个段落，同一个键可能来自邻文件的段落
    —— 所以只能「窗口 + 只保留文本确实出现在该文件里的行」，整文件比较会读到莫名其妙的 85%；
    ② 证据太薄（可比行 < 3）时判为「无法判断」，**不因此拒绝**（薄证据不构成否定）。
- **解析不到的种子只报告、绝不替换**（同 `run_path_selection._resolve_paths` 的 fail-fast 规则）。
  报告里出现本地找不到的文件时打 `GAP` 行，不猜、不指向同名的另一个文件。
- **子模块版本以 `git ls-tree` 为准，不靠版本号猜。** protobuf 的 abseil/googletest 是子模块：
  `git ls-tree HEAD third_party/` 给出报告生成时的那两个 pin（abseil `8c6e53e`、googletest `5ec7f0c`）。
  用 LTS 标签（20220623.x / release-1.11.0）会得到 2–6 行的偏移——正好被上面的对齐检查抓出来。

### 9.3 用法

```bash
# 按需（种子来自报告；默认写 pdg_<proj>.json + 预热 cfg_cache/<proj>/）
python3.10 tools/build_project_deps.py --project protobuf \
    --on-demand ~/csa_reports/project/protobuf/reports

# 先看清单不构建：种子 TU / 被调闭包 / 编译参数 / 语法通过率
python3.10 tools/build_project_deps.py --project protobuf --on-demand <dir> --dry-run

# 显式给 TU（不涉及报告）
python3.10 tools/build_project_deps.py --project foo --files a.cc b.cc

# 校验 PROJECT_SPECS 与 cfg_parser._CFG_PROJECT_CONFIGS 是否一致
python3.10 tools/build_project_deps.py --project protobuf --on-demand <dir> --check-config
```

流程：`[1] 种子 → [2] 被调闭包 → [3] 编译参数 → [4] 逐 TU -fsyntax-only 闸门 →`（非 dry-run）`[5] 写 compile db
→ [6] pdg_builder → [7] 预热 cfg_cache → [8] 汇总 JSON`。

可调项：`--closure-rounds`（默认 1，0 关闭）、`--closure-max-tus`（默认 25，硬上限）、`--min-line-match`（默认 0.9）、
`--compile-db`（默认 `/tmp/<proj>_cc/compile_commands.json`）、`--output`、`--no-cfg`、`--root`。

### 9.4 闭包宁可保守：过度闭包比不做更糟

第一版闭包用 `grep 'Name('`，29 个被调名拉进 **127 个 TU**（`compiler/csharp/*`、`compiler/java/*` 整片进来）——
因为 grep 命中的是**调用点、注释和其他 TU 里的同名函数**。现在改为 **ctags 定义索引**
（`ctags -x --c++-kinds=f -R`，一行一个真实函数定义，带 qualified name），再加三处收紧：
短名 < 4 字符或落在停用词里（`size`/`empty`/`find`…）的名不参与；qualified 被调名（`absl::Foo::Bar`）优先匹配带类名的定义；
每个名字最多取 2 个文件、闭包总量最多 25 个 TU。**头文件里的定义不加 TU**（包含它的 TU 已经涵盖）。

判据：闭包的价值是「报告确实走过的跨 TU 被调者」，不是「可能有关的代码」。后续项目若要放宽，
先看 `--dry-run` 的 `+ closure` 行逐条是否都对应报告里的一个具体调用。

### 9.5 protobuf 产物速查

| 项 | 值 |
|---|---|
| 源码树 | `~/csa_reports/project/protobuf`（270 MB；abseil `8c6e53e`、googletest `5ec7f0c` 按子模块 pin 拉取） |
| `compile_commands.json` | `/tmp/protobuf_cc/compile_commands.json`（21 TU，不入库） |
| `pdg_protobuf.json` | 仓库根 |
| `cfg_cache/protobuf/` | 21 个整-TU 缓存 |
| 工程配置 | `cfg_parser._CFG_PROJECT_CONFIGS["protobuf"]`（include: `.` / `src` / abseil / googletest+googlemock include；无 define） |
| 覆盖 | 21/21 份报告有种子（0 GAP）；21/21 TU 语法通过 |
| 待办 | 端到端冒烟（§6）后按需回填本项目判定层的表现 |
