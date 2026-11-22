"""15 个红灯测试：每个测试都直接调用尚未实现的桩函数。

测试只验证公开契约，不修改实现。浮点比较统一使用 1e-6 绝对容差，
所有信号都由确定性公式生成，不使用随机数。
"""

import math

import pytest

from granular.synth import (
    GranularError,
    grain_envelope,
    grain_schedule,
    granular_batch,
    overlap_add,
    pitch_jitter,
    read_grain,
    spray_position,
    validate_params,
)


def _tone(freq, seconds, sample_rate=1000, amp=0.5):
    """生成确定性的正弦测试信号，不依赖随机数。"""
    n = int(round(seconds * sample_rate))
    return [
        amp * math.sin(2.0 * math.pi * freq * i / sample_rate)
        for i in range(n)
    ]


def _config(**overrides):
    """批量合成的基础配置，测试按需覆盖单个字段。"""
    config = {
        "density": 4.0,
        "length_ms": 10.0,
        "sample_rate": 1000.0,
        "rate": 1.0,
        "seed": 20261119,
        "shape": "hann",
    }
    config.update(overrides)
    return config


def test_grain_schedule_uses_density_and_deterministic_jitter():
    """缺陷 1：颗粒起点不能是固定等间隔，density 与 seed 都必须生效。"""
    starts = grain_schedule(1000.0, 8.0, 20261119)

    assert len(starts) == 8
    assert starts[0] == pytest.approx(0.0, abs=1e-6)
    assert starts == sorted(starts)
    assert all(0.0 <= start < 1000.0 for start in starts)

    slots = [i * 1000.0 / 8.0 for i in range(8)]
    assert any(abs(start - slot) > 1e-6 for start, slot in zip(starts, slots))

    again = grain_schedule(1000.0, 8.0, 20261119)
    assert again == pytest.approx(starts, abs=1e-6)
    other = grain_schedule(1000.0, 8.0, 20261120)
    assert any(abs(a - b) > 1e-6 for a, b in zip(starts, other))


def test_grain_envelope_is_peak_normalized():
    """缺陷 2：包络必须归一到 1，不能整体放大后把叠加区推到削顶。"""
    hann = grain_envelope(101, "hann")
    assert len(hann) == 101
    assert hann[0] == pytest.approx(0.0, abs=1e-6)
    assert hann[-1] == pytest.approx(0.0, abs=1e-6)
    assert hann[50] == pytest.approx(1.0, abs=1e-6)
    assert max(hann) <= 1.0 + 1e-6

    tukey = grain_envelope(101, "tukey")
    assert len(tukey) == 101
    assert tukey[0] == pytest.approx(0.0, abs=1e-6)
    assert tukey[-1] == pytest.approx(0.0, abs=1e-6)
    assert tukey[50] == pytest.approx(1.0, abs=1e-6)
    assert max(tukey) <= 1.0 + 1e-6

    rect = grain_envelope(9, "rect")
    assert rect == [1.0] * 9


def test_read_grain_pads_out_of_bounds_with_zeros():
    """缺陷 3：读到源尾部之外必须补零，不能直接抛 IndexError。"""
    out = read_grain([1.0, 2.0, 3.0, 4.0], 3.0, 4.0, 1000.0, 1.0)

    assert len(out) == 4
    assert out == pytest.approx([4.0, 0.0, 0.0, 0.0], abs=1e-6)

    beyond = read_grain([1.0, 2.0, 3.0, 4.0], 10.0, 2.0, 1000.0, 1.0)
    assert beyond == pytest.approx([0.0, 0.0], abs=1e-6)


def test_overlap_add_normalizes_overlap_count():
    """缺陷 4：重叠相加必须除以重叠层数，不能整体变响或削顶。"""
    out = overlap_add([[1.0] * 4, [1.0] * 4], [0, 1], 4)

    assert len(out) == 4
    assert out == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=1e-6)
    assert max(abs(value) for value in out) <= 1.0 + 1e-6


def test_read_grain_interpolates_at_variable_rate():
    """缺陷 5：变速读取必须线性插值，不能取整造成台阶和咔哒声。"""
    slow = read_grain([0.0, 10.0, 20.0, 30.0], 0.0, 2.0, 1000.0, 0.5)
    assert slow == pytest.approx([0.0, 5.0], abs=1e-6)

    fast = read_grain([0.0, 10.0, 20.0, 30.0], 0.0, 3.0, 1000.0, 1.5)
    assert fast == pytest.approx([0.0, 15.0, 30.0], abs=1e-6)


def test_pitch_jitter_is_deterministic():
    """缺陷 6：音高抖动不能使用 random，同一 seed 必须完全可复现。"""
    value = pitch_jitter(440.0, 0.05, 12345)

    assert value == pytest.approx(pitch_jitter(440.0, 0.05, 12345), abs=1e-6)
    assert abs(value - 440.0) <= 440.0 * 0.05 + 1e-6
    assert pitch_jitter(440.0, 0.0, 7) == pytest.approx(440.0, abs=1e-6)

    values = [pitch_jitter(440.0, 0.05, seed) for seed in range(8)]
    assert len({round(item, 9) for item in values}) > 1


def test_spray_position_clamps_to_source_length():
    """缺陷 7：喷洒位置必须钳制在源长度内，不能越过尾部。"""
    source_ms = 1000.0
    base_pos = 995.0

    for seed in range(32):
        position = spray_position(source_ms, base_pos, seed)
        assert 0.0 <= position <= source_ms + 1e-6

    for seed in range(32):
        position = spray_position(source_ms, 5.0, seed)
        assert 0.0 <= position <= source_ms + 1e-6

    assert spray_position(source_ms, base_pos, 3) == pytest.approx(
        spray_position(source_ms, base_pos, 3), abs=1e-6
    )


def test_validate_params_reports_density_and_rate():
    """缺陷 8：density=0 必须被报告，不能进入会死循环的调度。"""
    issues = validate_params(
        {"length_ms": 20.0, "density": 0.0, "rate": 1.0}
    )
    assert len(issues) == 1
    assert "density" in issues[0].lower()
    assert validate_params(
        {"length_ms": 20.0, "density": 8.0, "rate": 1.0}
    ) == []

    rate_issues = validate_params(
        {"length_ms": 20.0, "density": 8.0, "rate": 0.0}
    )
    assert len(rate_issues) == 1
    assert "rate" in rate_issues[0].lower()


def test_granular_batch_isolates_sources():
    """缺陷 9：批量合成不能共用颗粒缓冲，静音源不能被其他源串音。"""
    voice = _tone(220.0, 1.0)
    silence = [0.0] * len(voice)
    config = _config(density=4.0, length_ms=10.0)

    solo = granular_batch({"voice": voice}, config)
    batch = granular_batch({"voice": voice, "silence": silence}, config)

    assert list(batch) == ["voice", "silence"]
    assert len(batch["voice"]) == len(solo["voice"])
    assert max(
        abs(a - b) for a, b in zip(batch["voice"], solo["voice"])
    ) <= 1e-6
    assert max(abs(value) for value in batch["silence"]) <= 1e-9


def test_zero_length_is_rejected_instead_of_dividing():
    """缺陷 10：length_ms=0 必须报参数问题并在读取入口抛 GranularError。"""
    issues = validate_params(
        {"length_ms": 0.0, "density": 4.0, "rate": 1.0}
    )
    assert len(issues) == 1
    assert "length_ms" in issues[0].lower()

    with pytest.raises(GranularError):
        read_grain([0.0, 0.1, 0.2], 0.0, 0.0, 1000.0, 1.0)


def test_single_grain_output_length_matches_grain_length():
    """组合：只调度出一个颗粒时，输出长度必须等于颗粒长度。"""
    source = [0.25] * 1000
    config = _config(density=1.0, length_ms=20.0, seed=7)

    out = granular_batch({"one": source}, config)["one"]

    assert len(out) == 20
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert max(abs(value) for value in out) <= 1.0 + 1e-6


def test_double_density_roughly_doubles_grain_count():
    """组合：density 翻倍时颗粒数应约翻倍，允许取整误差 ±1。"""
    sparse = grain_schedule(2000.0, 4.0, 99)
    dense = grain_schedule(2000.0, 8.0, 99)

    assert abs(len(dense) - 2 * len(sparse)) <= 1
    assert len(dense) > len(sparse)


def test_empty_samples_raise_granular_error():
    """组合：空样本必须抛 GranularError；批量中任一空源整体回滚。"""
    with pytest.raises(GranularError):
        read_grain([], 0.0, 10.0, 1000.0, 1.0)

    with pytest.raises(GranularError):
        granular_batch(
            {"good": [0.1] * 50, "empty": []},
            _config(),
        )


def test_rate_one_reads_samples_unchanged():
    """组合：rate=1 时读取结果必须与源片段逐采样相同（1e-6）。"""
    samples = [0.125, -0.25, 0.5, -0.75, 0.0]

    out = read_grain(samples, 0.0, 5.0, 1000.0, 1.0)

    assert len(out) == len(samples)
    assert out == pytest.approx(samples, abs=1e-6)


def test_granular_batch_preserves_source_order():
    """组合：批量全部成功时输出键顺序与输入完全一致。"""
    sources = {
        "first": _tone(220.0, 0.5),
        "second": _tone(330.0, 0.5),
        "third": _tone(440.0, 0.5),
    }
    config = _config()

    out = granular_batch(sources, config)

    assert list(out) == list(sources)
    assert all(len(out[name]) > 0 for name in sources)
    assert out["first"] == pytest.approx(
        granular_batch({"first": sources["first"]}, config)["first"], abs=1e-6
    )
    assert any(
        abs(a - b) > 1e-6
        for a, b in zip(out["first"], out["second"])
    )
