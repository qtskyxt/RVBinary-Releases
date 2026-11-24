"""Chinese and English UI text, including variable-bearing status messages."""

TRANSLATIONS = {
    ("RVBinary · 轨道生成器"): ("RVBinary · Orbit generator"),
    ("DM91 随机轨道实验室"): ("DM91 orbit laboratory"),
    ("轻量生成与可复现导出"): ("Lightweight sampling and reproducible export"),
    ("生成设置"): ("Generation settings"),
    ("主星质量"): ("Primary mass"),
    ("主星"): ("Primary star"),
    ("质量 / M☉"): ("Mass / M☉"),
    ("周期分布 · log10(P/day)"): ("Period distribution · log10(P/day)"),
    ("均值 μ"): ("Mean μ"),
    ("标准差 σ"): ("Standard deviation σ"),
    ("随机采样"): ("Random sampling"),
    ("轨道数量"): ("Orbit count"),
    ("随机种子"): ("Random seed"),
    ("生成轨道"): ("Generate orbits"),
    ("仅点击生成按钮时采样。相同参数和种子可复现结果。"): (
        "Sampling runs only when clicked. Identical parameters and seed reproduce the results."
    ),
    ("每条轨道使用相同主星质量和 logg，数量指独立轨道条数。"): (
        "Every orbit uses the same primary mass and logg. Count means independent orbits."
    ),
    ("输入参数无效："): ("Invalid parameters: "),
    ("正在生成轨道…"): ("Generating orbits…"),
    ("生成失败，未替换已有结果："): ("Generation failed; previous results retained: "),
    ("设置左侧参数并点击「生成轨道」，开始查看随机双星的分布与速度曲线。"): (
        "Set parameters in the sidebar and clic"
        "k Generate orbits to explore binary di"
        "stributions and RV curves."
    ),
    ("**一次生成，三种查看方式**：总体分布、单轨道 RV 曲线、完整参数表。"): (
        "**One batch, three views**: population"
        " distributions, individual RV curves, "
        "and orbital parameters."
    ),
    ("本界面生成轨道参数，不生成生产用 epoch/beta HDF5，也不执行 PDF/MCMC。"): (
        "This interface generates orbital param"
        "eters. Production epoch/beta HDF5 and "
        "PDF/MCMC are outside its scope."
    ),
    ("参数已修改，需重新生成。当前图表与下载仍对应上一次成功生成的参数。"): (
        "Parameters changed: regenerate to appl"
        "y them. Plots and downloads still use "
        "the last successful batch."
    ),
    ("已生成轨道"): ("Generated orbits"),
    ("周期模型 μ / σ"): ("Period model μ / σ"),
    ("采样耗时"): ("Sampling time"),
    ("总体分布"): ("Distributions"),
    ("单轨道 RV"): ("Single-orbit RV"),
    ("参数表与导出"): ("Parameters and export"),
    ("散点最多显示 2,000 条；导出包含所有轨道。"): (
        "Scatter shows at most 2,000 orbits; export contains the entire batch."
    ),
    (
        "采用现有 DM91Model 的质量比、偏心率和倾角采样，并保留恒星半径、近"
        "心点 Roche 约束与 log10(P/day) < 8 的截断。接受后的"
        "周期分布可能不同于输入的未截断正态分布。"
    ): (
        "Uses DM91Model mass-ratio, eccentricit"
        "y and inclination sampling, retaining "
        "stellar-radius and periastron Roche co"
        "nstraints and the log10(P/day) < 8 cut"
        "off. Accepted periods may differ from "
        "the input untruncated normal distribut"
        "ion."
    ),
    ("轨道 ID"): ("Orbit ID"),
    ("自定义时间跨度"): ("Custom time span"),
    ("时间跨度 / day"): ("Time span / day"),
    ("模拟观测与噪声"): ("Simulated observations and noise"),
    ("噪声标准差 / km/s"): ("Noise standard deviation / km/s"),
    ("观测点数"): ("Observation count"),
    ("噪声种子"): ("Noise seed"),
    ("观测点均匀覆盖时间跨度；独立高斯噪声，误差棒为 1σ。"): (
        "Observations uniformly cover the time "
        "span. Independent Gaussian noise; erro"
        "r bars show 1σ."
    ),
    ("下载含噪 RV CSV"): ("Download noisy RV CSV"),
    ("此轨道的 RV 曲线计算失败，参数表仍可导出："): (
        "RV calculation failed for this orbit; orbital parameters remain exportable: "
    ),
    (
        "默认两周期，512 个采样点；主星相对系统的速度，系统速度为零；叠加可复现的"
        "含噪观测点。M₀ 表示初始平近点角。高偏心率的近心点尖峰可能未被固定采样充分"
        "解析。"
    ): (
        "Default: two periods and 512 curve sam"
        "ples; primary RV relative to the syste"
        "m, with zero systemic velocity and rep"
        "roducible noisy observations. M₀ is th"
        "e initial mean anomaly. The fixed grid"
        " may underresolve high-eccentricity pe"
        "riastron peaks."
    ),
    ("预览前 "): ("Previewing first "),
    (" 条，共 "): (" rows of "),
    (" 条。角度单位为 rad。"): (" total. Angles are in radians."),
    ("下载完整参数 CSV"): ("Download full orbital CSV"),
    ("下载配置 JSON"): ("Download configuration JSON"),
    ("查看已生成参数快照"): ("Generated parameter snapshot"),
    ("orbit_id 是合成轨道索引，不是真实 TARGETID；CSV/JSON 不是生产流水线 mock 配置。"): (
        "orbit_id is a synthetic orbit index, n"
        "ot a real TARGETID. CSV/JSON are not p"
        "roduction mock configurations."
    ),
}


def translate(text: str, language: str) -> str:
    if language != "English":
        return text
    for chinese in sorted(TRANSLATIONS, key=len, reverse=True):
        text = text.replace(chinese, TRANSLATIONS[chinese])
    return text
