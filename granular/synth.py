"""颗粒合成核心：确定性颗粒排布、包络、变速读取、重叠相加与批量渲染。

约定
----
* 样本统一为 ``list[float]``，标称范围 ``-1.0 .. 1.0``；无限值、``NaN``
  或非法结构都视为非法输入。
* 所有时间参数（颗粒起点、颗粒长度、喷洒位置等）一律使用**毫秒（ms）**；
  ``sample_rate`` 使用 Hz。颗粒长度与输出长度按采样点精确换算。
* 颗粒位置、音高抖动和喷洒偏移全部由 ``seed`` 做确定性哈希，禁止使用
  ``random``；不联网、不 ``sleep``。
* 叠加时按每个采样点的重叠层数归一，不削顶、不整体变响。
* 浮点比较使用绝对容差 ``1e-6``；越界读取一律补 ``0.0``，不得抛
  ``IndexError``。
* 参数、源样本或批量配置非法时统一抛出 :class:`GranularError`。
"""

from __future__ import annotations

import math

_MASK64 = (1 << 64) - 1
_ENVELOPE_SHAPES = ("hann", "tukey", "rect")


class GranularError(ValueError):
    """颗粒合成参数、源样本或批量配置非法时抛出的异常。"""


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _is_int(value) -> bool:
    """``bool`` 虽然是 ``int`` 的子类，但不应被当作整数参数。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _to_finite_float(value):
    """把数值转成有限 ``float``；非法、``NaN``、``inf`` 时返回 ``None``。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _require_finite(value, name: str) -> float:
    result = _to_finite_float(value)
    if result is None:
        raise GranularError(f"{name} 必须是有限数值")
    return result


def _require_positive(value, name: str) -> float:
    result = _require_finite(value, name)
    if result <= 0.0:
        raise GranularError(f"{name} 必须大于 0")
    return result


def _require_seed(seed) -> int:
    if not _is_int(seed):
        raise GranularError("seed 必须是整数")
    return seed


def _mix64(value: int) -> int:
    """splitmix64 的确定性混合函数，不依赖平台随机化。"""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _hash_unit(seed: int, index: int) -> float:
    """把 ``(seed, index)`` 映射到 ``[0.0, 1.0)``，完全可复现。"""
    product = ((index & _MASK64) * 0x9E3779B97F4A7C15) & _MASK64
    combined = (seed & _MASK64) ^ product
    return _mix64(combined) / float(1 << 64)


def _validate_samples(samples, name: str = "samples") -> list[float]:
    """校验样本为非空、只含有限数值的序列，并返回浮点副本。"""
    if not isinstance(samples, (list, tuple)):
        raise GranularError(f"{name} 必须是数值列表")
    if len(samples) == 0:
        raise GranularError(f"{name} 不能为空")
    values: list[float] = []
    for value in samples:
        numeric = _to_finite_float(value)
        if numeric is None:
            raise GranularError(f"{name} 只能包含有限数值")
        values.append(numeric)
    return values


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def grain_schedule(duration_ms: float, density: float, seed: int) -> list[float]:
    """返回整段素材内的颗粒起点（ms），顺序递增且完全可复现。

    契约
    ----
    * ``duration_ms > 0``、``density > 0``（单位：颗粒/秒）、``seed`` 为
      整数；否则抛 :class:`GranularError`。
    * 颗粒数
      ``n = max(1, floor(duration_ms * density / 1000.0 + 0.5))``；
      ``density`` 翻倍时 ``n`` 约翻倍（允许取整造成 ±1 的差异）。
    * 第 0 个颗粒固定在 ``0.0 ms``；第 ``i`` 个颗粒的标称槽位为
      ``i * 1000.0 / density``，在该槽位附近用 ``(seed, i)`` 的确定性
      哈希做不超过半个槽宽的抖动。不得使用 ``random``。
    * 返回值按升序排列，全部落在 ``[0, duration_ms)`` 内；同一组参数多次
      调用必须逐项相同（容差 ``1e-6``），不同 ``seed`` 应产生不同位置。
    * 不修改输入，不返回 ``NaN`` 或无限值。
    """

    duration = _require_positive(duration_ms, "duration_ms")
    dens = _require_positive(density, "density")
    seed = _require_seed(seed)

    count = max(1, int(math.floor(duration * dens / 1000.0 + 0.5)))
    slot_width = 1000.0 / dens
    # 保证最后一个槽位的抖动后位置仍严格小于 duration_ms。
    upper_limit = math.nextafter(duration, 0.0)

    starts: list[float] = []
    for i in range(count):
        if i == 0:
            starts.append(0.0)
            continue
        unit = _hash_unit(seed, i)
        offset = (unit - 0.5) * slot_width
        position = i * slot_width + offset
        if position < 0.0:
            position = 0.0
        elif position >= duration:
            position = upper_limit
        starts.append(position)

    starts.sort()
    return starts


def grain_envelope(length: int, shape: str) -> list[float]:
    """生成单个颗粒的包络，长度精确为 ``length`` 个采样，峰值归一为 1。

    契约
    ----
    * ``length`` 为 ``>= 1`` 的整数，``shape`` 只接受 ``"hann"``、
      ``"tukey"``、``"rect"``；否则抛 :class:`GranularError`。
    * 返回 ``list[float]``，长度为 ``length``，每个值落在 ``0.0 .. 1.0``，
      峰值不超过 ``1.0 + 1e-6``，不得整体放大。
    * ``"hann"``：``0.5 - 0.5 * cos(2*pi*i/(length-1))``，两端为 ``0``，
      中间峰值 ``1``；``length == 1`` 时返回 ``[1.0]``。
    * ``"tukey"``：固定 ``alpha = 0.5`` 的平顶余弦窗，两端为 ``0``，
      中间平台为 ``1``；``length == 1`` 时返回 ``[1.0]``。
    * ``"rect"``：所有采样均为 ``1.0``，是唯一端点不为 ``0`` 的形状。
    * 端点、峰值和平台比较均使用绝对容差 ``1e-6``。
    """

    if not _is_int(length) or length < 1:
        raise GranularError("length 必须是 >= 1 的整数")
    if not isinstance(shape, str) or shape not in _ENVELOPE_SHAPES:
        raise GranularError("shape 只能是 'hann'、'tukey' 或 'rect'")

    if length == 1:
        return [1.0]

    if shape == "rect":
        return [1.0] * length

    if shape == "hann":
        denominator = length - 1
        return [
            0.5 - 0.5 * math.cos(2.0 * math.pi * i / denominator)
            for i in range(length)
        ]

    # Tukey（alpha = 0.5）：两端余弦过渡，中间为幅度 1 的平台。
    alpha = 0.5
    n_minus_1 = length - 1
    edge = alpha * n_minus_1 / 2.0
    flat_end = n_minus_1 - edge
    result: list[float] = []
    for n in range(length):
        if n < edge:
            value = 0.5 * (1.0 + math.cos(math.pi * (n / edge - 1.0)))
        elif n > flat_end:
            value = 0.5 * (
                1.0 + math.cos(math.pi * (n / edge - 2.0 / alpha + 1.0))
            )
        else:
            value = 1.0
        # 余弦的浮点误差可能让结果略微越界。
        result.append(min(1.0, max(0.0, value)))
    return result


def read_grain(
    samples: list[float],
    start_ms: float,
    length_ms: float,
    sample_rate: float,
    rate: float,
) -> list[float]:
    """从 ``samples`` 按 ``rate`` 变速读取一个颗粒。

    契约
    ----
    * ``samples`` 必须非空且只含有限数值；``start_ms >= 0``、
      ``length_ms > 0``、``sample_rate > 0``、``rate > 0``，否则抛
      :class:`GranularError`。``length_ms == 0`` 必须被拒绝，不能进入除法。
    * 输出采样数
      ``out_len = max(1, floor(length_ms * sample_rate / 1000.0 + 0.5))``。
    * 第 ``i`` 个输出对应源位置
      ``p = start_ms * sample_rate / 1000.0 + i * rate``；
      在 ``floor(p)`` 与 ``ceil(p)`` 之间做线性插值，不能只用整数下标，
      否则变速时会出现台阶和咔哒声。浮点误差不超过 ``1e-6``。
    * 越界部分按 ``0.0`` 补齐：``p < 0`` 或 ``p >= len(samples)`` 时，
      缺失的一侧视为 ``0.0``，不得抛 ``IndexError``；颗粒可以从源尾部
      之外开始并自然淡出。
    * ``rate == 1`` 时，从整数毫秒位置读取的结果必须与源片段逐采样相同
      （绝对容差 ``1e-6``）。
    * 返回新列表，不修改 ``samples``。
    """

    values = _validate_samples(samples)
    start = _require_finite(start_ms, "start_ms")
    if start < 0.0:
        raise GranularError("start_ms 必须 >= 0")
    length = _require_positive(length_ms, "length_ms")
    sr = _require_positive(sample_rate, "sample_rate")
    speed = _require_positive(rate, "rate")

    out_len = max(1, int(math.floor(length * sr / 1000.0 + 0.5)))
    base = start * sr / 1000.0
    count = len(values)

    result: list[float] = []
    for i in range(out_len):
        position = base + i * speed
        index = math.floor(position)
        fraction = position - index
        left = values[index] if 0 <= index < count else 0.0
        right = values[index + 1] if 0 <= index + 1 < count else 0.0
        result.append(left + (right - left) * fraction)
    return result


def overlap_add(
    grains: list[list[float]],
    positions: list[int],
    total_len: int,
) -> list[float]:
    """把多个颗粒按采样位置叠加，并按重叠层数归一，避免削顶。

    契约
    ----
    * ``grains`` 与 ``positions`` 等长；``positions[i]`` 是 ``>= 0`` 的
      整数采样偏移；``total_len`` 为 ``>= 1`` 的整数；否则抛
      :class:`GranularError`。
    * 输出长度精确为 ``total_len``；超出输出末尾的颗粒部分直接截断。
    * 对输出采样 ``t``：
      ``out[t] = sum(grain[t - positions[i]]) / max(1, coverage[t])``，
      其中 ``coverage[t]`` 是覆盖该点的颗粒层数（即包络覆盖数）。
      不能只把重叠样本相加，否则重叠区会整体变响甚至削顶。
    * 只要输入颗粒采样在 ``-1.0 .. 1.0`` 内，输出也必须落在
      ``-1.0 .. 1.0``；相同信号完全重叠时保持原幅度。
    * 空 ``grains`` 返回 ``total_len`` 个 ``0.0``；输入不被修改。
    * 浮点比较使用绝对容差 ``1e-6``。
    """

    if not isinstance(grains, (list, tuple)):
        raise GranularError("grains 必须是列表")
    if not isinstance(positions, (list, tuple)):
        raise GranularError("positions 必须是列表")
    if len(grains) != len(positions):
        raise GranularError("grains 与 positions 长度必须相等")
    if not _is_int(total_len) or total_len < 1:
        raise GranularError("total_len 必须是 >= 1 的整数")
    for position in positions:
        if not _is_int(position) or position < 0:
            raise GranularError("positions 必须是非负整数")

    output = [0.0] * total_len
    coverage = [0] * total_len

    # 每个颗粒独立累加，绝不共享或复用其他颗粒/源的缓冲。
    for grain, position in zip(grains, positions):
        if not isinstance(grain, (list, tuple)):
            raise GranularError("grains 的每个元素都必须是数值列表")
        for offset, value in enumerate(grain):
            numeric = _to_finite_float(value)
            if numeric is None:
                raise GranularError("颗粒采样只能包含有限数值")
            target = position + offset
            if target >= total_len:
                break
            output[target] += numeric
            coverage[target] += 1

    for t in range(total_len):
        if coverage[t] > 0:
            output[t] /= coverage[t]
    return output


def pitch_jitter(freq: float, spread: float, seed: int) -> float:
    """返回一个颗粒的确定性音高，围绕 ``freq`` 做比例抖动。

    契约
    ----
    * ``freq > 0``、``spread >= 0``、``seed`` 为整数；否则抛
      :class:`GranularError`。
    * 返回 ``freq * (1.0 + offset)``，其中
      ``offset`` 由 ``seed`` 的确定性哈希映射到
      ``[-spread, +spread]``。不得使用 ``random`` 或读取系统时间。
    * 同一组 ``(freq, spread, seed)`` 多次调用必须返回相同结果
      （绝对容差 ``1e-6``）；不同 ``seed`` 应产生不同音高。
    * ``spread == 0`` 时精确返回 ``freq``。结果必须是有限正数。
    """

    base_freq = _require_positive(freq, "freq")
    amount = _require_finite(spread, "spread")
    if amount < 0.0:
        raise GranularError("spread 必须 >= 0")
    seed = _require_seed(seed)

    if amount == 0.0:
        return base_freq

    unit = _hash_unit(seed, 0)
    offset = (2.0 * unit - 1.0) * amount
    factor = 1.0 + offset
    if factor <= 0.0:
        # 极端 spread 下仍保证结果是有限正数。
        factor = math.nextafter(0.0, 1.0)
    return base_freq * factor


def spray_position(base_ms: float, spray_ms: float, seed: int) -> float:
    """返回颗粒的确定性喷洒位置（ms），绝不超过源素材长度。

    约定
    ----
    为了保持题目给定的三参数签名，本函数约定：``base_ms`` 是源素材的
    长度（ms），也是返回值上界；``spray_ms`` 是未经抖动的基准位置（ms）。

    契约
    ----
    * ``base_ms > 0``、``spray_ms >= 0``、``seed`` 为整数；否则抛
      :class:`GranularError`。
    * 位置 = ``spray_ms + offset``，``offset`` 由 ``seed`` 的确定性哈希
      产生，幅度不超过 ``min(20.0, 0.1 * base_ms)`` 毫秒。
    * 随后把位置严格钳制到 ``[0.0, base_ms]``，无论基准位置离源尾部多近，
      结果都不能超出源长度；不得使用 ``random``。
    * 同一组参数多次调用结果相同（容差 ``1e-6``），不同 ``seed`` 应产生
      不同位置；返回值是有限浮点数。
    """

    source_ms = _require_positive(base_ms, "base_ms")
    base_position = _require_finite(spray_ms, "spray_ms")
    if base_position < 0.0:
        raise GranularError("spray_ms 必须 >= 0")
    seed = _require_seed(seed)

    amplitude = min(20.0, 0.1 * source_ms)
    unit = _hash_unit(seed, 0)
    position = base_position + (2.0 * unit - 1.0) * amplitude

    if position < 0.0:
        position = 0.0
    elif position > source_ms:
        position = source_ms
    return position


def validate_params(params: dict) -> list[str]:
    """检查颗粒参数，返回问题描述列表；本函数不抛异常。

    契约
    ----
    * ``params`` 应为 ``dict``，至少包含 ``length_ms``、``density``、
      ``rate`` 三个键；缺少键、非数值、``NaN`` 或无限值都记为问题。
    * ``length_ms <= 0`` 记为问题，避免后续按长度换算时除零。
    * ``density <= 0`` 记为问题，``density == 0`` 时不得进入颗粒调度循环
      （否则会死循环）。
    * ``rate <= 0`` 记为问题。
    * 每个问题返回一个字符串，且必须包含对应字段名（``length_ms``、
      ``density`` 或 ``rate``）；顺序与上述检查顺序一致。
    * 全部合法时返回 ``[]``。额外字段忽略；``params`` 不是 ``dict`` 时
      返回单条问题描述，而不是抛异常。
    * 浮点比较使用绝对容差 ``1e-6``；本函数只读取 ``params``，不修改输入。
    """

    if not isinstance(params, dict):
        return ["params 必须是 dict，且包含 length_ms、density、rate"]

    issues: list[str] = []
    for field in ("length_ms", "density", "rate"):
        if field not in params:
            issues.append(f"缺少字段 {field}")
            continue
        value = _to_finite_float(params[field])
        if value is None:
            issues.append(f"{field} 必须是有限数值")
            continue
        if value <= 1e-6:
            issues.append(f"{field} 必须大于 0")
    return issues


def granular_batch(sources: dict, config: dict) -> dict:
    """批量合成颗粒，每个源独立渲染；任一源非法则整体失败并回滚。

    契约
    ----
    * ``sources`` 为 ``{名称: list[float]}``，保持插入顺序；每组样本必须
      非空、只含有限数值，标称范围 ``-1.0 .. 1.0``。
    * ``config`` 必须包含 ``density``、``length_ms``、``sample_rate``、
      ``rate``、``seed``、``shape``。长度、密度和速率由
      :func:`validate_params` 校验；``sample_rate > 0``，``shape`` 只能是
      ``"hann"``、``"tukey"``、``"rect"``。
    * 先对所有源和整个配置做整体校验。任一源为空、越界、含 ``NaN``/
      ``inf``，或配置非法，立即抛 :class:`GranularError`，不得跳过坏源，
      也不得返回部分结果（整体回滚）。
    * 每个源使用自己独立的颗粒缓冲，源与源之间不得串音，也不得复用未
      清空的缓冲。
    * 单个源的渲染方式：以 ``len(samples) / sample_rate * 1000`` 毫秒为
      时长调用 :func:`grain_schedule`；对每个起点用 :func:`read_grain`
      读取 ``length_ms`` 的颗粒，乘以长度
      ``floor(length_ms * sample_rate / 1000 + 0.5)`` 的
      :func:`grain_envelope`，再用 :func:`overlap_add` 叠加。
    * 每个源输出缓冲的长度取最后一个颗粒的结束位置（至少 1 个采样）；
      只调度出一个从 ``0 ms`` 开始的颗粒时，输出长度恰等于颗粒长度。
    * 返回 ``{名称: list[float]}``，键顺序与 ``sources`` 完全一致；成功时
      每个源都必须出现，且每个输出采样都在 ``-1.0 .. 1.0`` 内。
    * 输出长度与数值比较使用绝对容差 ``1e-6``；不得修改 ``sources`` 或
      ``config``。
    """

    if not isinstance(sources, dict):
        raise GranularError("sources 必须是 dict")
    if not isinstance(config, dict):
        raise GranularError("config 必须是 dict")

    required = ("density", "length_ms", "sample_rate", "rate", "seed", "shape")
    for key in required:
        if key not in config:
            raise GranularError(f"config 缺少字段 {key}")

    issues = validate_params(config)
    if issues:
        raise GranularError("; ".join(issues))

    sample_rate = _require_positive(config["sample_rate"], "sample_rate")
    seed = _require_seed(config["seed"])
    shape = config["shape"]
    if not isinstance(shape, str) or shape not in _ENVELOPE_SHAPES:
        raise GranularError("shape 只能是 'hann'、'tukey' 或 'rect'")

    density = float(config["density"])
    length_ms = float(config["length_ms"])
    rate = float(config["rate"])

    # 先整体校验全部源，确保任一坏源都会让整次批量渲染回滚。
    validated_sources: dict[str, list[float]] = {}
    for name, samples in sources.items():
        values = _validate_samples(samples, f"sources[{name!r}]")
        for value in values:
            if value < -1.0 - 1e-6 or value > 1.0 + 1e-6:
                raise GranularError(f"sources[{name!r}] 采样超出 -1.0 .. 1.0")
        validated_sources[name] = values

    grain_len = max(1, int(math.floor(length_ms * sample_rate / 1000.0 + 0.5)))
    envelope = grain_envelope(grain_len, shape)

    rendered: dict[str, list[float]] = {}
    for name, samples in validated_sources.items():
        duration_ms = len(samples) / sample_rate * 1000.0
        starts = grain_schedule(duration_ms, density, seed)

        grains: list[list[float]] = []
        positions: list[int] = []
        for start_ms in starts:
            grain = read_grain(samples, start_ms, length_ms, sample_rate, rate)
            # read_grain 与 grain_len 使用同一取整公式，长度天然一致；
            # 仍按较短者截断以防御未来改动。
            windowed = [
                grain[i] * envelope[i] for i in range(min(len(grain), grain_len))
            ]
            grains.append(windowed)
            positions.append(int(math.floor(start_ms * sample_rate / 1000.0 + 0.5)))

        # 最后一个颗粒的结束位置；starts 升序，因此它就是缓冲总长。
        total_len = max(1, positions[-1] + grain_len)
        rendered[name] = overlap_add(grains, positions, total_len)

    return rendered
