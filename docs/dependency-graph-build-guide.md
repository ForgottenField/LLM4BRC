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
[compile_commands.json]  ──►  (2) pdg_builder_local  ──►  pdg_<proj>.json   （放仓库根）
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
- **(2)** 由 C++ 工具 `pdg_builder_local` 完成，命令行固定 `--compile-db <db> --output <out>`。
- **(3)** 复用 `cfg_feasibility.build_cfg_command/run_cfg_dump/parse_cfg_text/_save_cfg_cache`，逐个源文件跑一次 `clang++-14 ... debug.DumpCFG`。

核心洞察：**只要 (a) 拿到一套正确的 compile_commands.json，(b) 能在 `clang++-14 -std=c++17` 下语法通过，(c) 登记工程配置——三件套即可对任意 C++ 项目离线生成**，且 (2)/(3) 与具体项目无关、只需参数。

---

## 2. 步骤一：生成 `compile_commands.json`（最易踩坑的一步）

PDG 工具 `pdg_builder_local` 走 `tooling::JSONCompilationDatabase`，即把每条编译指令交给它去喂给 clang AST。它只认 **clang 能真正编译通过的 TU**；任何编不过的 TU 要么剔除、要么单独修。

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
src/llm_client/csa_analysis/pdg/build/pdg_builder_local \
  --compile-db /tmp/faiss_cc/compile_commands.json \
  --output pdg_faiss.json
```

**验证三件事**：
1. 文件生成、体积合理（对照：aria2 46MB/24019 fn、folly 21MB/10840 fn、faiss 37MB/13648 fn）。
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
bash src/llm_client/csa_analysis/pdg/build.sh   # 产出 build/pdg_builder_local
```

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
| ① 报告宿主的**源文件清单**（建哪些 CFG） | ② `pdg_builder_local` 调用（固定 CLI） |
| ③ **include / define / force_include**（进 compile_commands 与 `_CFG_PROJECT_CONFIGS`） | ④ `build_cfg_command/run_cfg_dump/parse_cfg_text/_save_cfg_cache` 全套 |
| ⑤ 手写 compile_commands 时**需要额外系统头/宏**（omp.h、FINTEGER 之类） | ⑥ `load_sdg` / md5 缓存命名 / 顶层查找逻辑 |
| ⑦ **源码根路径** `project_root` / `source_root` | ⑧ 判空/建目录/验证等胶水 |

### 7.1 推荐的统一形态：`config.py + 一个 build 脚本`

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
3. `pdg_builder_local --compile-db ... --output pdg_<proj>.json`。
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
