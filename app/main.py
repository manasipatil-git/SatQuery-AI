import os, glob
import numpy as np, streamlit as st
from PIL import Image
import analysis as A

DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demo")
st.set_page_config("SatQuery AI", layout="wide", page_icon="🛰️")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400&display=swap');
.stApp{background:#0a0c0f;background-image:linear-gradient(rgba(34,211,238,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.035) 1px,transparent 1px);background-size:48px 48px;font-family:Inter,sans-serif;color:#d7dde4}
header[data-testid=stHeader]{background:transparent}.block-container{padding-top:1rem;max-width:1500px}
section[data-testid=stSidebar]{background:#0d1014;border-right:1px solid #1d242c}
.top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1d242c;padding:6px 0 14px;margin-bottom:14px}
.brand{font-weight:600;letter-spacing:.28em;font-size:1.15rem}.tag{color:#7b8794;font-size:.8rem;font-weight:300}
.mono{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#7b8794;letter-spacing:.08em}.ok{color:#22d3ee}
.panel{background:rgba(18,23,29,.72);backdrop-filter:blur(8px);border:1px solid #232c36;border-radius:6px;padding:14px 16px;margin-bottom:12px}
.h{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.18em;color:#22d3ee;margin-bottom:8px}
.big{font-size:1.15rem;font-weight:600}.bar{height:4px;background:#1b222a;border-radius:2px;margin:3px 0 9px}.bar>div{height:4px;border-radius:2px}
.row{display:flex;justify-content:space-between;font-size:.82rem}.warn{color:#fb923c}
.stButton>button{background:#10151b;border:1px solid #2a3440;color:#cfd6dd;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.06em}
.stButton>button:hover{border-color:#22d3ee;color:#22d3ee}
.stTextInput input{background:#0f1419;border:1px solid #2a3440;color:#e6edf3;border-radius:4px}
img{border-radius:4px;border:1px solid #232c36}
</style>""", unsafe_allow_html=True)
S = st.session_state
HELP = "This score measures how clearly pixels separate between clusters. It is not validation accuracy."
for k, v in dict(res=None, name="—", ans=None, chg=None, up=None, ca=None, cb=None, q="", meta=None, demo=False, qual=None).items(): S.setdefault(k, v)

def rd(f):
    try: return A.load_image(f)[0]
    except A.ImageError as e: st.error(str(e)); return None

def load(f, name, demo=False):
    try: im, meta = A.load_image(f)
    except A.ImageError as e: st.error(str(e)); return
    S.ans = None; S.name = name; S.meta = meta; S.demo = demo; S.qual = q = A.quality(im)
    if q["severe"]: S.res = None; st.error("IMAGE QUALITY WARNING — " + " ".join(q["warnings"]) + " Analysis skipped to avoid misleading output."); return
    try:
        with st.spinner("◌ ANALYZING…"): S.res = A.analyze(im)
    except Exception as e: S.res = None; st.error(f"Analysis failed: {e}")

with st.sidebar:
    st.markdown('<div class="h">IMAGE INPUT</div>', unsafe_allow_html=True)
    f = st.file_uploader("Drop satellite image", ["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
    if f and S.up != (f.name, f.size): S.up = (f.name, f.size); load(f, f.name)
    st.markdown('<div class="h">DEMO DATASET</div>', unsafe_allow_html=True)
    demos = sorted(p for p in glob.glob(os.path.join(DEMO, "*.*")) if not os.path.basename(p).startswith("change_"))
    pick = st.selectbox("Scene", demos, format_func=lambda p: os.path.basename(p).split(".")[0].replace("_", " ").upper(), label_visibility="collapsed") if demos else None
    if st.button("LOAD DEMO IMAGE", disabled=not demos): load(pick, os.path.basename(pick), demo=True)
    with st.expander("MODEL & METHOD"):
        st.markdown("`INPUT` → `PREPROCESS` (resize ≤800px) → `FEATURES` (ExG, G−B, R−B, HSV, Laplacian texture) → `ML` (K-means, k=6, scikit-learn; no deep learning) → `CLASSIFY` (cluster→class rules) → `VISUALIZE` (overlays, contours, heatmap) → `NL ANSWER` (keyword intent router over masks).\n\nChange: ECC alignment → CIELAB difference → threshold + morphology → class transitions. Estimated classification, not a trained semantic model.")

with st.sidebar:
    with st.expander("RECOMMENDED DEMO"):
        st.markdown("1. Load real satellite image\n2. Run segmentation\n3. Ask “Where are the water bodies?”\n4. Show highlighted evidence\n5. Ask “Where is urban development concentrated?”\n6. Switch to FEATURES\n7. Open COMPARE\n8. Run change detection")
res = S.res; meta = S.meta
mlbl = "DEMO" if S.demo else "REAL RGB"
info = (f'IMAGE <b>{S.name}</b> &nbsp;│&nbsp; RESOLUTION <b>{meta["width"]}×{meta["height"]}</b> → ANALYSIS <b>{meta["proc"][0]}×{meta["proc"][1]}</b> &nbsp;│&nbsp; MODE <b class="{"warn" if S.demo else "ok"}">{mlbl}</b> &nbsp;│&nbsp; TIME <b>{res["time"]:.2f}s</b>' if res and meta else 'IMAGE <b>—</b> &nbsp;│&nbsp; MODE <b>—</b>')
st.markdown(f"""<div class="top"><div><span class="brand">SATQUERY AI</span> &nbsp;<span class="tag">Talk to the Earth. · Geospatial analysis</span></div>
<div class="mono">{info} &nbsp;│&nbsp; <span class="ok">● {'ANALYSIS READY' if res else 'AWAITING IMAGE'}</span></div></div>""", unsafe_allow_html=True)
if res and meta:
    atype = "SYNTHETIC DEMO SCENE" if S.demo else ("REAL SATELLITE / AERIAL RGB" if meta["kind"].startswith("RGB") else meta["kind"].upper())
    st.caption(f"ANALYSIS TYPE: {atype} · {meta['mode']} ({meta['fmt']}) · aspect {meta['aspect']} · Spectral indices unavailable for RGB imagery.")
    if S.qual and S.qual["warnings"]: st.warning("IMAGE QUALITY WARNING — " + " ".join(S.qual["warnings"]))
mode = st.segmented_control("mode", ["ORIGINAL", "SEGMENTATION", "WATER", "VEGETATION", "BUILT-UP", "HEATMAP", "FEATURES", "COMPARE"], default="ORIGINAL", label_visibility="collapsed") or "ORIGINAL"

if mode == "COMPARE":
    st.markdown('<div class="h">CHANGE DETECTION</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    ua = c1.file_uploader("2023 / IMAGE A", ["png", "jpg", "jpeg", "tif", "tiff"]); ub = c2.file_uploader("2026 / IMAGE B", ["png", "jpg", "jpeg", "tif", "tiff"])
    if c3.button("LOAD DEMO PAIR"):
        S.ca, S.cb = rd(os.path.join(DEMO, "change_2023.png")), rd(os.path.join(DEMO, "change_2026.png"))
    if ua: S.ca = rd(ua)
    if ub: S.cb = rd(ub)
    if c3.button("ANALYZE CHANGE", type="primary"):
        if S.ca is None or S.cb is None: st.warning("Provide both images (or load the demo pair).")
        else:
            try:
                with st.spinner("Registering and differencing…"): S.chg = A.change(S.ca, S.cb)
            except Exception as e: st.error(f"Change analysis failed: {e}")
    g = S.chg
    if g:
        x = st.columns(3)
        x[0].image(g["a"], caption="IMAGE A · BEFORE"); x[1].image(g["b"], caption="IMAGE B · AFTER"); x[2].image(g["overlay"], caption="CHANGE OVERLAY")
        y = st.columns([1, 1, 1.4]); y[0].image(g["diff"], caption="DIFFERENCE MAP")
        y[1].markdown(f'<div class="panel"><div class="h">VISIBLE CHANGE</div><div class="big warn">{g["pct"]:.1f}%</div><div class="mono">{g["regions"]} significant regions · alignment {"ECC ok" if g["aligned"] else "resize only"}<br>MEAN DIFFERENCE (LAB) {g["mean_diff"]:.0f}<br>LARGEST {(str(round(g["largest"]["area"], 1)) + "% · " + g["largest"]["where"]) if g["largest"] else "—"}</div></div>', unsafe_allow_html=True)
        rows = "".join(f'<div class="row"><span>{n}</span><span class="{"warn" if s_ != "Not detected" else "mono"}">{s_} · {v:.1f}%</span></div>' for n, v, s_ in g["items"])
        regs = "".join(f'<li>{r["where"]} · {r["area"]:.1f}% · {r["label"]}</li>' for r in g["regs"][:6])
        y[2].markdown(f'<div class="panel"><div class="h">CHANGE REPORT</div>{rows}<div class="h" style="margin-top:10px">TOP REGIONS</div><ul style="font-size:.78rem;padding-left:16px">{regs or "<li>No significant regions</li>"}</ul><div class="mono warn">{g["warn"]}</div></div>', unsafe_allow_html=True)
elif not res:
    st.markdown('<div class="panel" style="text-align:center;padding:90px 20px"><div class="big">No image loaded</div><div class="mono">UPLOAD A SATELLITE IMAGE OR CLICK “LOAD DEMO IMAGE” IN THE SIDEBAR</div></div>', unsafe_allow_html=True)
else:
    left, right = st.columns([3, 1.25]); img, ans = res["img"], S.ans
    with left:
        z = st.columns([1, 1, 1, 3]); show = z[0].toggle("Overlays", True)
        z[1].button("FIT", on_click=lambda: S.update(zoom=100)); z[2].button("RESET", on_click=lambda: S.update(zoom=100, ans=None))
        zoom = z[3].slider("Zoom %", 50, 200, 100, key="zoom", label_visibility="collapsed")
        sel = None
        if mode == "FEATURES" and res["feats"]: sel = st.selectbox("Select detected feature", [f["name"] for f in res["feats"]])
        if mode == "SEGMENTATION": out = A.seg_overlay(res)
        elif mode in ("WATER", "VEGETATION", "BUILT-UP"): out = A.class_view(res, mode.lower())
        elif mode == "HEATMAP":
            hc = st.selectbox("Heatmap class", ["auto (answer / built-up)"] + A.CLASSES[:4]); auto = hc.startswith("auto")
            out = A.heat_overlay(img, ans["heat"] if auto and ans and ans.get("heat") is not None else res["heat"] if auto else A._heat(res["cls"] == A.CLASSES.index(hc)))
        elif mode == "FEATURES": out = A.feat_overlay(img, res["feats"], sel)
        else: out = img
        if show and ans and ans.get("mask") is not None and mode != "HEATMAP": out = A.highlight(out, ans["mask"])
        if zoom == 100: st.image(out)
        else: st.image(out, width=int(out.shape[1] * zoom / 100))
        if mode == "SEGMENTATION":
            st.markdown("".join(f'<span class="mono" style="margin-right:16px"><span style="color:rgb{A.RGB[c]}">■</span> {c.upper()}</span>' for c in A.CLASSES), unsafe_allow_html=True)
        if mode == "HEATMAP": st.markdown('<span class="mono">HEATMAP = spatial density of the ' + ("queried class" if ans and ans.get("heat") is not None else "built-up / dominant class") + ' mask</span>', unsafe_allow_html=True)
        if sel:
            f_ = next(f for f in res["feats"] if f["name"] == sel)
            st.markdown(f'<div class="panel"><div class="h">FEATURE</div><div class="row"><span>{f_["name"]}</span><span class="mono">{f_["cls"].upper()} · {f_["conf"]:.0f}% · {f_["where"]} · {f_["area"]:.1f}% of scene · bbox {f_["box"]} · centroid {f_["centroid"]}</span></div></div>', unsafe_allow_html=True)
        if mode == "FEATURES" and res["feats"]:
            st.dataframe([dict(Feature=f["name"], Class=f["cls"], Area=f"{f['area']:.1f}%", Centroid=str(f["centroid"]), BBox=str(f["box"]), Location=f["where"], SeparationScore=f"{f['conf']:.0f}%") for f in res["feats"]], hide_index=True)
        st.markdown('<div class="h" style="margin-top:10px">ASK THE EARTH</div>', unsafe_allow_html=True)
        qs = ["Where are the water bodies?", "What is the dominant land cover?", "Where is vegetation concentrated?", "Are there signs of urban development?", "What major features can you identify?"]
        cols = st.columns(len(qs)); asked = None
        for c, q in zip(cols, qs):
            if c.button(q): asked = q
        with st.form("ask", clear_on_submit=True):
            t = st.text_input("q", placeholder="Ask anything about this image…", label_visibility="collapsed")
            if st.form_submit_button("ASK") and t.strip(): asked = t
        if asked: S.ans = A.answer(asked, res, S.chg); S.q = asked; st.rerun()
    with right:
        fr = res["frac"]
        bars = "".join(f'<div class="row"><span>{c.title()}</span><span class="mono">{fr[c]:.1f}%</span></div><div class="bar"><div style="width:{fr[c]}%;background:rgb{A.RGB[c]}"></div></div>' for c in A.CLASSES)
        st.markdown(f'<div class="panel"><div class="h">AI SCENE BRIEF</div><div class="mono">ENVIRONMENT</div><div class="big">{res["env"]}</div><br>{bars}<div class="mono">ESTIMATED LAND-COVER COMPOSITION (HEURISTIC, NOT GROUND TRUTH)<br>DOMINANT {res["top"].upper()} · WATER REGIONS {res["counts"]["water"]} · VEGETATION REGIONS {res["counts"]["vegetation"]} · BUILT-UP REGIONS {res["counts"]["built-up"]}<br><span title="{HELP}">CLUSTER SEPARATION SCORE ⓘ {res["overall"]:.0f}%</span> (not accuracy)</div></div>', unsafe_allow_html=True)
        if ans:
            fx = "".join(f'<div class="row"><span class="mono">{a_}</span><span>{b_}</span></div>' for a_, b_ in ans.get("facts", []))
            st.markdown(f'<div class="panel"><div class="h">AI ANALYSIS · “{S.q}”</div><div class="big">{ans["title"]}</div><p style="font-size:.85rem">{ans["text"]}</p>{fx}<div class="h">EVIDENCE</div><div class="mono">{ans["evid"]}</div><br><div class="h" title="{HELP}">CLUSTER SEPARATION SCORE ⓘ</div><div class="big">{ans["conf"]:.0f}%</div></div>', unsafe_allow_html=True)
        kf = "".join(f"<li>{f['name']} — {f['where']} ({f['conf']:.0f}%)</li>" for f in res["feats"][:5]) or "<li>No significant regions</li>"
        st.markdown(f'<div class="panel"><div class="h">WHAT I SEE</div><p style="font-size:.85rem">{res["brief"]}</p><div class="h">KEY FEATURES</div><ul style="font-size:.8rem;padding-left:16px">{kf}</ul></div>', unsafe_allow_html=True)


if res and mode != "COMPARE":
    m_ = S.meta; fr = res["frac"]; ct = res["counts"]
    with st.expander("HOW SATQUERY AI ANALYZES THIS IMAGE"):
        st.markdown(f"""**ML ALGORITHM: K-MEANS CLUSTERING** · **CLUSTERS: K = {res["k"]}** · **FEATURE DIMENSIONS: {res["dims"]}** · **CLASSIFICATION TYPE: UNSUPERVISED**

1. **Image preprocessing** — RGB, resized to {m_["proc"][0]}×{m_["proc"][1]} (from {m_["width"]}×{m_["height"]}), scaled to [0,1]
2. **Feature extraction** — {", ".join(A.FEATURES)}
3. **Feature normalization** — z-score (zero mean, unit variance) per feature
4. **K-means clustering** — fitted on 15,000 sampled pixels; every pixel then assigned to its nearest centroid
5. **Cluster-to-class mapping** — rules on each cluster's mean colour / ExG / saturation / texture / edge statistics (heuristic naming, not learned labels); ambiguous pixels → *other*
6. **Connected-component analysis** — morphology, then labelled regions ≥ 0.3% of the scene
7. **Spatial reasoning** — region centroid → 8-point compass
8. **Natural-language query** — keyword intent router picks the mask; the answer is read from the computed region statistics""")
    with st.expander("MODEL VALIDATION"):
        st.markdown(f"""**VALIDATION DATASET:** Not configured · **Accuracy:** Not reported · **Precision:** Not reported · **Recall:** Not reported · **F1:** Not reported

*Reason: the current pipeline is unsupervised and does not have ground-truth labels for quantitative validation.*

**System metrics** — processing time **{res["time"]:.2f}s** · clusters **{res["k"]}** · detected regions: water **{ct["water"]}**, vegetation **{ct["vegetation"]}**, built-up **{ct["built-up"]}** · estimated class shares: {", ".join(f"{c} {fr[c]:.1f}%" for c in A.CLASSES)} · cluster separation score **{res["overall"]:.1f}%** (geometric measure, not accuracy)""")
