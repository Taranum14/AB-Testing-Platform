import altair as alt
import pandas as pd
import streamlit as st

from src.simulate import simulate_experiment
from src.stats_tests import welch_ttest, sample_size_per_group
from src.cuped import cuped_adjust, variance_reduction
from src.sequential import msprt_path, first_stop
from src.peeking_simulation import peeking_false_positive_rate

st.set_page_config(page_title="Experiment analysis", layout="wide")

INK, MUTED, GRID = "#14181f", "#6b7280", "#e5e7eb"
RAW_C, CUPED_C, ALERT_C = "#8d96a3", "#0f62fe", "#da1e28"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1180px; }
h1 { font-size: 1.3rem !important; font-weight: 600 !important; letter-spacing: -0.01em; padding: 0 0 .25rem 0; }
h3 { font-size: .95rem !important; font-weight: 600 !important; margin-top: 1.5rem; }
.sub { color: #6b7280; font-size: .85rem; margin-bottom: 1.2rem; }
.kpi { border-top: 2px solid #14181f; padding-top: .55rem; margin-bottom: 1rem; }
.kpi .l { font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; color: #6b7280; }
.kpi .v { font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem; font-weight: 500; margin: .15rem 0; }
.kpi .s { font-size: .78rem; color: #6b7280; }
.stTabs [data-baseweb="tab"] { font-size: .88rem; }
.note { font-size: .82rem; color: #6b7280; border-left: 2px solid #e5e7eb; padding-left: .75rem; margin: .8rem 0; }
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub=""):
    return (f'<div class="kpi"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def style(chart):
    return (chart.configure_view(stroke=None)
            .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelColor=MUTED, titleColor=MUTED, labelFont="IBM Plex Sans",
                            titleFont="IBM Plex Sans", titleFontWeight="normal")
            .configure_legend(labelColor=INK, titleColor=MUTED, orient="top", title=None))


# ---------------- sidebar ----------------
st.sidebar.markdown("**Data**")
source = st.sidebar.radio("Source", ["Simulated", "Upload CSV"], label_visibility="collapsed")

if source == "Upload CSV":
    up = st.sidebar.file_uploader("Columns: group (0/1), metric, pre_metric", type="csv")
    if up is None:
        st.title("Experiment analysis")
        st.markdown('<div class="sub">Upload a CSV with columns <code>group</code> (0 = control, '
                    '1 = treatment), <code>metric</code> and <code>pre_metric</code>.</div>',
                    unsafe_allow_html=True)
        st.stop()
    df = pd.read_csv(up)
    missing = {"group", "metric", "pre_metric"} - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(sorted(missing))}")
        st.stop()
    df["group"] = df["group"].astype(int)
else:
    n_users = st.sidebar.number_input("Users", 2_000, 200_000, 20_000, step=2_000)
    rho = st.sidebar.slider("Pre / post correlation", 0.0, 0.95, 0.70, 0.05)
    lift = st.sidebar.slider("True relative lift", 0.0, 0.10, 0.03, 0.005, format="%.3f")
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
    df = simulate_experiment(n=int(n_users), rho=rho, lift=lift, seed=int(seed))

st.sidebar.markdown("**Analysis**")
alpha = st.sidebar.select_slider("Significance level", [0.01, 0.05, 0.10], value=0.05)
tau = st.sidebar.number_input("Sequential prior scale (tau)", 0.05, 10.0, 1.5, step=0.25)
mde = st.sidebar.number_input("Minimum effect to detect (absolute)", 0.1, 50.0, 1.0, step=0.5)

# ---------------- computation ----------------
g = df["group"].values
y_adj, theta = cuped_adjust(df["metric"], df["pre_metric"])
df["metric_cuped"] = y_adj
raw = welch_ttest(df["metric"][g == 0], df["metric"][g == 1], alpha)
cup = welch_ttest(df["metric_cuped"][g == 0], df["metric_cuped"][g == 1], alpha)
ctrl_mean = df["metric"][g == 0].mean()
vr = variance_reduction(df["metric"].values, y_adj)
n_raw = sample_size_per_group(df["metric"].std(), mde, alpha)
n_cup = sample_size_per_group(pd.Series(y_adj).std(), mde, alpha)

st.title("Experiment analysis")
st.markdown(f'<div class="sub">{len(df):,} users &middot; {int((g==0).sum()):,} control / '
            f'{int((g==1).sum()):,} treatment &middot; &alpha; = {alpha}</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Fixed horizon", "Sequential monitoring", "Peeking check"])

# ---------------- tab 1 ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Lift, raw", f"{raw['diff']/ctrl_mean:+.2%}", f"p = {raw['p_value']:.2g}"), unsafe_allow_html=True)
    c2.markdown(kpi("Lift, CUPED", f"{cup['diff']/ctrl_mean:+.2%}", f"p = {cup['p_value']:.2g}"), unsafe_allow_html=True)
    c3.markdown(kpi("Variance removed", f"{vr:.1%}", f"theta = {theta:.3f}"), unsafe_allow_html=True)
    c4.markdown(kpi("Users / arm needed", f"{n_cup:,}", f"{n_raw:,} without CUPED"), unsafe_allow_html=True)

    st.markdown("### Effect estimate with 95% interval" if alpha == 0.05
                else f"### Effect estimate with {1-alpha:.0%} interval")
    est = pd.DataFrame({"method": ["Raw", "CUPED"],
                        "diff": [raw["diff"], cup["diff"]],
                        "lo": [raw["ci_low"], cup["ci_low"]],
                        "hi": [raw["ci_high"], cup["ci_high"]]})
    colors = alt.Scale(domain=["Raw", "CUPED"], range=[RAW_C, CUPED_C])
    base = alt.Chart(est).encode(y=alt.Y("method:N", title=None, sort=None))
    bars = base.mark_rule(strokeWidth=4).encode(
        x=alt.X("lo:Q", title="Absolute difference in metric"), x2="hi:Q",
        color=alt.Color("method:N", scale=colors, legend=None))
    pts = base.mark_point(filled=True, size=110, opacity=1).encode(
        x="diff:Q", color=alt.Color("method:N", scale=colors, legend=None))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x="x:Q")
    st.altair_chart(style((zero + bars + pts).properties(height=130, width="container")))

    st.markdown("### Estimates")
    table = pd.DataFrame({
        "Method": ["Raw", "CUPED"],
        "Difference": [f"{raw['diff']:.3f}", f"{cup['diff']:.3f}"],
        "Std. error": [f"{raw['se']:.3f}", f"{cup['se']:.3f}"],
        "Interval": [f"[{raw['ci_low']:.3f}, {raw['ci_high']:.3f}]", f"[{cup['ci_low']:.3f}, {cup['ci_high']:.3f}]"],
        "Interval width": [f"{raw['ci_high']-raw['ci_low']:.3f}", f"{cup['ci_high']-cup['ci_low']:.3f}"],
        "p-value": [f"{raw['p_value']:.3g}", f"{cup['p_value']:.3g}"],
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.markdown('<div class="note">The adjusted estimate targets the same effect as the raw one; '
                'it is just measured with less noise, so the interval is narrower.</div>',
                unsafe_allow_html=True)

# ---------------- tab 2 ----------------
with tab2:
    step = max(50, len(df) // 400)
    ns_c, _, p_c = msprt_path(df["metric_cuped"][g == 0].values, df["metric_cuped"][g == 1].values, tau, step=step)
    ns_r, _, p_r = msprt_path(df["metric"][g == 0].values, df["metric"][g == 1].values, tau, step=step)
    stop_c, stop_r = first_stop(ns_c, p_c, alpha), first_stop(ns_r, p_r, alpha)

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Decision point, raw", f"{stop_r:,}" if stop_r else "none", "users per arm"), unsafe_allow_html=True)
    c2.markdown(kpi("Decision point, CUPED", f"{stop_c:,}" if stop_c else "none", "users per arm"), unsafe_allow_html=True)
    if stop_r and stop_c:
        c3.markdown(kpi("Earlier by", f"{1 - stop_c/stop_r:.0%}", "with CUPED"), unsafe_allow_html=True)

    path = pd.concat([
        pd.DataFrame({"n": ns_r, "p": p_r.clip(1e-10), "method": "Raw"}),
        pd.DataFrame({"n": ns_c, "p": p_c.clip(1e-10), "method": "CUPED"})])
    ysc = alt.Scale(type="log", domain=[1e-8, 1])
    line = alt.Chart(path).mark_line(strokeWidth=2).encode(
        x=alt.X("n:Q", title="Users per arm"),
        y=alt.Y("p:Q", scale=ysc, title="Always-valid p-value"),
        color=alt.Color("method:N", scale=colors))
    rule = alt.Chart(pd.DataFrame({"y": [alpha]})).mark_rule(color=ALERT_C, strokeDash=[5, 4]).encode(
        y=alt.Y("y:Q", scale=ysc))
    st.markdown("### Always-valid p-value over time")
    st.altair_chart(style((line + rule).properties(height=330, width="container")))
    st.markdown('<div class="note">The p-value never increases, so it is safe to check this chart every '
                'day and stop once it crosses the dashed line. Experiment is conclusive at that point '
                'without any peeking penalty.</div>', unsafe_allow_html=True)

# ---------------- tab 3 ----------------
@st.cache_data(show_spinner="Running 1,000 A/A experiments")
def run_peeking(looks, tau_):
    return peeking_false_positive_rate(n_looks=looks, tau=tau_)


with tab3:
    st.markdown("Simulates 1,000 experiments where there is **no real effect** and checks how often each "
                "rule declares a winner. The target is the significance level.")
    looks = st.slider("Number of interim looks", 2, 50, 20)
    res = run_peeking(looks, 0.1)
    rates = pd.DataFrame({"rule": ["Single look", f"Peek {looks} times", f"Sequential, {looks} looks"],
                          "rate": list(res.values())})
    bars = alt.Chart(rates).mark_bar(size=46).encode(
        x=alt.X("rule:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title="False positive rate", axis=alt.Axis(format="%")),
        color=alt.Color("rule:N", legend=None, scale=alt.Scale(
            range=[RAW_C, ALERT_C, CUPED_C])))
    labels = bars.mark_text(dy=-8, font="IBM Plex Mono", color=INK).encode(text=alt.Text("rate:Q", format=".1%"))
    target = alt.Chart(pd.DataFrame({"y": [0.05]})).mark_rule(color=INK, strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(style((bars + labels + target).properties(height=330, width="container")))