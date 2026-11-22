# audio-granular-synth

## 题面

给氛围音乐接入颗粒合成后，颗粒要么糊成一片、要么整段削顶：颗粒起点按等间隔排布，density 不生效；包络不归一，多个颗粒叠加后削顶；读取越过源尾部时直接报错而不是补零；重叠相加不除包络和，整体变响；变速读取用整数插值，出现咔哒；音高抖动用 random，结果不可复现；喷洒位置不钳制，超出源长度；参数不校验，density=0 时死循环；批量共用颗粒缓冲，源之间串音；length_ms 为 0 不校验，除零。granular/synth.py 是核心，tests/test_synth.py 15个红。要修的是：把上述 10 个缺陷逐一修掉，保持颗粒位置确定可复现、包络与叠加不削顶、越界读取补零、参数校验统一抛 GranularError，并让 15 个测试全部转绿。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始（未实现）状态下应恰好输出 `15 failed`，且没有 collection error、import error 或 syntax error：

```text
15 failed
```

每个测试都直接调用 `granular/synth.py` 中的桩函数，因此当前失败原因是 `NotImplementedError`。实现完成后应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 目录结构

```text
audio-granular-synth/
├── README.md
├── conftest.py
├── .gitignore
├── granular/
│   ├── __init__.py
│   └── synth.py
└── tests/
    └── test_synth.py
```

`conftest.py` 将仓库根加入 `sys.path`，因此测试无需安装包即可导入 `granular`。实现只使用标准库 `math`，不联网、不使用随机数、不调用 `sleep`。

## 接口

`granular/synth.py` 提供 8 个待实现函数：

| 函数 | 作用 |
| --- | --- |
| `grain_schedule(duration_ms, density, seed)` | 返回确定性的颗粒起点（ms），`density` 决定颗粒数 |
| `grain_envelope(length, shape)` | 生成归一化的 `hann` / `tukey` / `rect` 包络 |
| `read_grain(samples, start_ms, length_ms, sample_rate, rate)` | 按变速比例线性插值读取，越界补零 |
| `overlap_add(grains, positions, total_len)` | 按重叠层数归一叠加，不削顶 |
| `pitch_jitter(freq, spread, seed)` | 用确定性哈希做颗粒音高抖动 |
| `spray_position(base_ms, spray_ms, seed)` | 返回钳制在源长度内的喷洒位置 |
| `validate_params(params)` | 返回长度、密度、速率等问题列表 |
| `granular_batch(sources, config)` | 批量合成，任一源非法时整体回滚且不串音 |

## 约束

- 样本统一为 `list[float]`，标称范围 `-1.0 .. 1.0`；时间参数均以毫秒（ms）为单位，`sample_rate` 为 Hz。
- 颗粒长度、输出长度按采样精确换算；浮点比较容差为 `1e-6`。
- 颗粒位置、音高抖动、喷洒偏移只允许使用基于 `seed` 的确定性哈希，禁止 `random`；不联网、不 `sleep`。
- 异常统一使用 `granular.synth.GranularError`。
- `tests/test_synth.py` 不可修改；只允许修改 `granular/synth.py`（必要时可调整 `granular/__init__.py` 的导出）。
