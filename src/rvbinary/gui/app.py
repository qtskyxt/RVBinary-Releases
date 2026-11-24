"""Streamlit page for a lightweight, independent DM91 orbit generator."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from rvbinary import __version__
from rvbinary.gui.plots import (
    distribution_figure,
    observations_to_csv,
    period_eccentricity_figure,
    rv_figure,
    rv_observations,
)
from rvbinary.gui.state import parameters_changed
from rvbinary.gui.translations import translate
from rvbinary.mock.generator import (
    GeneratorConfig,
    batch_config_json,
    batch_to_csv,
    generate_orbits,
    orbit_row,
)
from rvbinary.models.base import SimulationError


def show_figure(figure):
    try:
        st.pyplot(figure, width="stretch")
    finally:
        plt.close(figure)


def main() -> None:
    st.set_page_config(page_title="RVBinary", page_icon="✦", layout="wide")
    language = st.sidebar.selectbox("语言 / Language", ["中文", "English"], key="language")

    def t(text):
        return translate(text, language)

    st.title(t("RVBinary · 轨道生成器"))
    st.caption(t(f"DM91 随机轨道实验室 · RVBinary {__version__} · 轻量生成与可复现导出"))

    with st.sidebar:
        st.header(t("生成设置"))
        st.subheader(t("主星"))
        mass = st.number_input(t("质量 / M☉"), value=1.0, step=0.1, format="%.4f", key="mass")
        logg = st.number_input("logg / cgs", value=4.5, step=0.1, key="logg")
        st.subheader(t("周期分布 · log10(P/day)"))
        mean = st.number_input(t("均值 μ"), value=4.8, step=0.1, key="mean")
        sigma = st.number_input(t("标准差 σ"), value=2.3, step=0.1, key="sigma")
        st.subheader(t("随机采样"))
        count = st.number_input(
            t("轨道数量"),
            min_value=1,
            max_value=10000,
            value=1000,
            step=100,
            key="count",
        )
        seed = st.number_input(
            t("随机种子"),
            min_value=0,
            max_value=4294967295,
            value=20261010,
            step=1,
            key="seed",
        )
        clicked = st.button(t("生成轨道"), key="generate", type="primary", width="stretch")
        st.caption(t("仅点击生成按钮时采样。相同参数和种子可复现结果。"))
        st.caption(t("每条轨道使用相同主星质量和 logg，数量指独立轨道条数。"))

    config = None
    try:
        config = GeneratorConfig(count, seed, mass, logg, mean, sigma)
    except ValueError as exc:
        st.error(t(f"输入参数无效：{exc}"))

    if clicked and config is not None:
        try:
            started = perf_counter()
            with st.spinner(t("正在生成轨道…")):
                new_batch = generate_orbits(config)
            st.session_state["batch"] = new_batch
            st.session_state["generation_seconds"] = perf_counter() - started
            st.session_state["orbit_id"] = 0
            st.session_state.pop("span_days", None)
        except (SimulationError, ValueError, OverflowError, FloatingPointError) as exc:
            st.error(t(f"生成失败，未替换已有结果：{exc}"))

    batch = st.session_state.get("batch")
    if batch is None:
        st.info(t("设置左侧参数并点击「生成轨道」，开始查看随机双星的分布与速度曲线。"))
        st.markdown(t("**一次生成，三种查看方式**：总体分布、单轨道 RV 曲线、完整参数表。"))
        st.caption(t("本界面生成轨道参数，不生成生产用 epoch/beta HDF5，也不执行 PDF/MCMC。"))
        return

    if config is None or parameters_changed(config, batch):
        st.warning(t("参数已修改，需重新生成。当前图表与下载仍对应上一次成功生成的参数。"))
    columns = st.columns(4)
    columns[0].metric(t("已生成轨道"), f"{len(batch.orbits):,}")
    columns[1].metric(t("随机种子"), str(batch.config.seed))
    columns[2].metric(
        t("周期模型 μ / σ"),
        f"{batch.config.log10_period_mean:g} / {batch.config.log10_period_sigma:g}",
    )
    columns[3].metric(t("采样耗时"), f"{st.session_state['generation_seconds']:.2f} s")
    st.caption(t(f"主星质量 {batch.config.mass_solar:g} M☉ · logg {batch.config.logg_cgs:g} cgs"))

    distribution, curve, table = st.tabs([t("总体分布"), t("单轨道 RV"), t("参数表与导出")])
    with distribution:
        show_figure(distribution_figure(batch))
        show_figure(period_eccentricity_figure(batch))
        st.caption(t("散点最多显示 2,000 条；导出包含所有轨道。"))
        st.info(
            t(
                "采用现有 DM91Model 的质量比、偏心率和倾角采样，并保留恒星半径、"
                "近心点 Roche 约束与 log10(P/day) < 8 的截断。"
                "接受后的周期分布可能不同于输入的未截断正态分布。"
            )
        )
    with curve:
        selected = st.number_input(
            t("轨道 ID"),
            min_value=0,
            max_value=len(batch.orbits) - 1,
            value=0,
            step=1,
            key="orbit_id",
        )
        orbit = batch.orbits[selected]
        st.caption(
            f"P = {orbit.period_days:.6g} day · q = {orbit.mass_ratio:.4f} · "
            f"e = {orbit.eccentricity:.4f} · i = {np.degrees(orbit.inclination_rad):.2f}° · "
            f"ω = {np.degrees(orbit.argument_periastron_rad):.2f}° · "
            f"M₀ = {np.degrees(orbit.mean_anomaly_rad):.2f}°"
        )
        span = None
        if st.toggle(t("自定义时间跨度"), key="custom_span"):
            span = st.number_input(
                t("时间跨度 / day"),
                value=2.0 * orbit.period_days,
                key="span_days",
                format="%.6g",
            )
        st.subheader(t("模拟观测与噪声"))
        noise_sigma = st.number_input(
            t("噪声标准差 / km/s"), min_value=0.0, value=1.0, step=0.1, key="noise_sigma"
        )
        observation_count = st.number_input(
            t("观测点数"), min_value=2, max_value=10000, value=50, step=1, key="observation_count"
        )
        noise_seed = st.number_input(
            t("噪声种子"),
            min_value=0,
            max_value=4294967295,
            value=20261010,
            step=1,
            key="noise_seed",
        )
        st.caption(t("观测点均匀覆盖时间跨度；独立高斯噪声，误差棒为 1σ。"))
        try:
            observations = rv_observations(orbit, span, observation_count, noise_sigma, noise_seed)
            show_figure(rv_figure(orbit, span, observations=observations))
            st.download_button(
                t("下载含噪 RV CSV"),
                observations_to_csv(*observations),
                f"rvbinary_rv_orbit_{selected}_seed_{noise_seed}.csv",
                "text/csv; charset=utf-8",
                key="download_rv",
                on_click="ignore",
            )
        except (SimulationError, ValueError, OverflowError, FloatingPointError) as exc:
            st.error(t(f"此轨道的 RV 曲线计算失败，参数表仍可导出：{exc}"))
        st.caption(
            t(
                "默认两周期，512 个采样点；主星相对系统的速度，"
                "系统速度为零；叠加可复现的含噪观测点。"
                "M₀ 表示初始平近点角。高偏心率的近心点尖峰可能未被固定采样充分解析。"
            )
        )
    with table:
        st.dataframe([orbit_row(o) for o in batch.orbits[:100]], hide_index=True, width="stretch")
        st.caption(
            t(
                f"预览前 {min(100, len(batch.orbits))} 条，共 {len(batch.orbits):,} 条。"
                "角度单位为 rad。"
            )
        )
        left, right = st.columns(2)
        left.download_button(
            t("下载完整参数 CSV"),
            batch_to_csv(batch),
            "rvbinary_orbits.csv",
            "text/csv; charset=utf-8",
            key="download_csv",
            on_click="ignore",
        )
        right.download_button(
            t("下载配置 JSON"),
            batch_config_json(batch),
            "rvbinary_generator_config.json",
            "application/json",
            key="download_json",
            on_click="ignore",
        )
        with st.expander(t("查看已生成参数快照")):
            st.json(asdict(batch.config))
        st.caption(
            t("orbit_id 是合成轨道索引，不是真实 TARGETID；CSV/JSON 不是生产流水线 mock 配置。")
        )


if __name__ == "__main__":
    main()
