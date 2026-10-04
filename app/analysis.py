"""SatQuery AI analysis core: unsupervised ML (K-means on spectral+texture features) + CV."""
import time
import numpy as np, cv2
from PIL import Image
from sklearn.cluster import KMeans

CLASSES = ["water", "vegetation", "built-up", "bare land", "other"]
RGB = {"water": (56, 189, 248), "vegetation": (74, 222, 128), "built-up": (251, 191, 36), "bare land": (176, 141, 108), "other": (100, 110, 125)}
INTENT = {"water": ["water", "river", "lake", "sea", "ocean", "coast"],
          "vegetation": ["vegetation", "green", "forest", "tree", "farm", "crop", "agricult"],
          "built-up": ["urban", "built", "building", "city", "develop", "road", "settle", "infrastructure"], "bare land": ["bare", "soil", "barren", "desert"]}

def prep(a, m=800):
    h, w = a.shape[:2]; s = m / max(h, w)
    return cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else a

class ImageError(Exception): pass
MAX_PIXELS = 150_000_000
Image.MAX_IMAGE_PIXELS = None  # we enforce our own limit

def load_image(src):
    """Safe loader -> (rgb uint8 resized for computation, metadata). Raises ImageError with a friendly message."""
    try:
        im = Image.open(src); w, h = im.size; fmt, mode0 = im.format or "?", im.mode
        if min(w, h) < 32: raise ImageError("Image is too small to analyse (minimum 32 px per side).")
        if w * h > MAX_PIXELS: raise ImageError(f"Image is too large ({w}×{h}). Please crop or downscale to under ~150 megapixels.")
        im.draft("RGB", (2400, 2400)) if fmt == "JPEG" else None
        if im.mode in ("I;16", "I", "F"):   # 16-bit / single-band TIFF -> 2-98 percentile stretch to 8 bit
            a = np.array(im).astype(np.float32); lo, hi = np.percentile(a, (2, 98))
            im = Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8))
        arr = np.array(im.convert("RGB"))
    except ImageError: raise
    except Exception: raise ImageError("This image format is not supported (or the file is corrupt). Please upload PNG, JPG, JPEG, or an RGB / single-band TIFF. Multispectral TIFFs are not supported.")
    gray = mode0 in ("L", "LA", "I;16", "I", "F", "1")
    out = prep(arr)
    meta = dict(width=w, height=h, aspect=f"{w / h:.2f}", fmt=fmt, mode=mode0, proc=(out.shape[1], out.shape[0]),
                kind="Grayscale/single-band (converted to RGB)" if gray else "RGB satellite/aerial image",
                note="Single-band image converted to RGB: colour-based classes (water/vegetation) are unreliable." if gray else
                     "RGB imagery only: no NIR/SWIR bands, so NDVI/NDWI/NDBI are NOT computed.")
    return out, meta

FEATURES = ["Red", "Green", "Blue", "Excess Green (ExG)", "Green−Blue", "Red−Blue", "HSV saturation", "HSV value", "Local texture (Laplacian)", "Edge density (Canny)", "Local variance"]

def quality(rgb):
    """Simple input-quality indicators; 'severe' => analysis would be misleading."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    q = dict(brightness=float(g.mean()), contrast=float(g.std()), saturation=float(hsv[..., 1].mean() / 2.55),
             edge_density=float((cv2.Canny(g, 60, 150) > 0).mean() * 100), near_black=float((rgb.max(2) < 20).mean() * 100), near_white=float((rgb.min(2) > 235).mean() * 100))
    w = []
    if q["brightness"] < 45: w.append("Image is very dark.")
    if q["contrast"] < 20: w.append("Low contrast detected. Classification confidence may be reduced.")
    if q["near_black"] > 30: w.append(f"{q['near_black']:.0f}% near-black pixels (no-data border or shadow).")
    if q["near_white"] > 30: w.append(f"{q['near_white']:.0f}% near-white pixels (cloud, snow or saturation).")
    q["warnings"] = w; q["severe"] = q["contrast"] < 5 or q["near_black"] > 90 or q["near_white"] > 90 or q["brightness"] < 15
    return q

def _feat(rgb):  # 11 engineered features per pixel
    f = rgb.astype(np.float32) / 255; r, g, b = f[..., 0], f[..., 1], f[..., 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1), cv2.CV_32F)), (0, 0), 4) * 10
    ed = cv2.GaussianBlur((cv2.Canny((gray * 255).astype(np.uint8), 60, 150) > 0).astype(np.float32), (0, 0), 4) * 4
    lv = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2, 0)) * 4
    return np.stack([r, g, b, 2 * g - r - b, g - b, r - b, hsv[..., 1] / 255, hsv[..., 2] / 255, tex, ed, lv], -1)

def segment(rgb, k=6, seed=0):
    F = _feat(rgb); H, W, D = F.shape; F = F.reshape(-1, D)
    s = np.random.RandomState(seed).choice(len(F), min(15000, len(F)), replace=False)
    mu, sd = F[s].mean(0), F[s].std(0) + 1e-6; Z = (F - mu) / sd          # z-score feature normalisation
    km = KMeans(k, n_init=3, random_state=seed).fit(Z[s])                 # unsupervised K-means
    d = km.transform(Z); lab = d.argmin(1); ds = np.sort(d, 1)
    conf = np.clip((ds[:, 1] - ds[:, 0]) / (ds[:, 1] + 1e-6), 0, 1).reshape(H, W)   # cluster separation score
    cnt = np.bincount(lab, minlength=k); C = np.stack([np.bincount(lab, weights=F[:, j], minlength=k) / np.maximum(cnt, 1) for j in range(D)], 1)  # raw-unit centroids
    ed_med, tx_med = np.median(C[:, 9]), np.median(C[:, 8]); name = {}
    for c in range(k):   # heuristic cluster -> class naming from centroid statistics
        R_, G_, B_, exg, gb, rb, sat, val, tex, ed, lv = C[c]
        name[c] = ("other" if cnt[c] == 0 else "vegetation" if (exg > 0.1 and G_ >= R_) else
                   "water" if (rb < -0.08 or (rb < -0.02 and tex <= tx_med and ed <= ed_med) or (val < 0.22 and rb < 0.03 and tex <= tx_med)) else
                   "built-up" if (sat < 0.2 or (sat < 0.35 and ed >= ed_med and tex >= tx_med)) else "bare land")
    lut = np.array([CLASSES.index(name[c]) for c in range(k)]); cls = lut[lab].reshape(H, W)
    cls[conf < 0.08] = 4; cls[rgb.max(2) < 8] = 4   # 'other' = ambiguous pixels and pure-black no-data
    frac = {c: float((cls == i).mean() * 100) for i, c in enumerate(CLASSES)}
    return dict(img=rgb, cls=cls, conf=conf, frac=frac, k=k, dims=D, mconf={c: float(conf[cls == i].mean()) if (cls == i).any() else 0 for i, c in enumerate(CLASSES)},
                overall=float(conf.mean() * 100))

def region(c, W, H):  # 8-point compass + CENTER from centroid position
    x, y = c[0] / W - .5, c[1] / H - .5
    if abs(x) < .17 and abs(y) < .17: return "CENTER"
    return "-".join(p for p in ("NORTH" if y < -.17 else "SOUTH" if y > .17 else "", "WEST" if x < -.17 else "EAST" if x > .17 else "") if p)

def comps(mask, conf, n=8):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    k, l, st, cen = cv2.connectedComponentsWithStats(m); H, W = m.shape; out = []
    for i in range(1, k):
        if st[i, 4] < 0.003 * H * W: continue
        out.append(dict(box=tuple(int(v) for v in st[i, :4]), area=float(st[i, 4] / (H * W) * 100),
                        where=region(cen[i], W, H), centroid=(int(cen[i][0]), int(cen[i][1])), conf=float(conf[l == i].mean() * 100)))
    out.sort(key=lambda d: -d["area"]); return out[:n], m.astype(bool)

def _heat(mask):
    h = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(mask.shape) / 25); return h / (h.max() + 1e-6)

def analyze(img):
    t0 = time.perf_counter(); r = segment(img); feats = []; counts = {}
    for i, c in enumerate(CLASSES[:3]):
        f, _ = comps(r["cls"] == i, r["conf"], 999); counts[c] = len(f)
        for x in f: x["cls"] = c
        feats += f[:8]
    feats.sort(key=lambda d: -d["area"]); feats = feats[:10]
    for i, f in enumerate(feats): f["name"] = f"{f['cls'].title()} region {i + 1}"
    fr = r["frac"]; top = max(CLASSES[:4], key=fr.get)
    env = ("Coastal / Urban" if fr["water"] > 15 and fr["built-up"] > 15 else "Urban" if fr["built-up"] > 35 else
           "Agricultural / Vegetated" if fr["vegetation"] > 40 else "Water-dominated" if fr["water"] > 40 else "Mixed")
    hm = r["cls"] == 2 if fr["built-up"] > 3 else r["cls"] == CLASSES.index(top)
    r.update(feats=feats, counts=counts, env=env, heat=_heat(hm), top=top, time=time.perf_counter() - t0,
             brief=f"{env} terrain. {top.title()} dominates ({fr[top]:.0f}%); {len(feats)} distinct regions found.")
    return r

def _where(fs):
    tot = sum(f["area"] for f in fs); d = {}
    for f in fs: d[f["where"]] = d.get(f["where"], 0) + f["area"]
    top = [k for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if v >= 0.25 * tot][:2]
    return " and ".join(top) if top else fs[0]["where"]

def answer(q, res, chg=None):
    ql = q.lower(); R = dict(mask=None, heat=None, conf=res["overall"])
    if "change" in ql or "differ" in ql:
        if not chg: return dict(R, title="NO COMPARISON YET", text="Load two images in COMPARE and click ANALYZE CHANGE first.", evid="Change detection has not been run.")
        det = "; ".join(f"{n}: {v:.1f}%" for n, v, s_ in chg["items"] if s_ != "Not detected")
        return dict(R, title=f"VISIBLE CHANGE {chg['pct']:.1f}%", text=f"{chg['regions']} significant regions. " + (det or "No class transition above threshold; semantic transition uncertain."), evid="CIELAB pixel difference + per-region class transitions (see COMPARE).")
    c = next((k for k, v in INTENT.items() if any(w in ql for w in v)), None); dom = False
    if c is None and any(w in ql for w in ("dominant", "most", "cover", "composition")): c = max(CLASSES[:4], key=res["frac"].get); dom = True
    if c is None:
        t = "; ".join(f"{f['name']} ({f['where']}, {f['area']:.1f}%)" for f in res["feats"][:4])
        return dict(R, title="SCENE OVERVIEW", text=res["brief"] + (" Main features: " + t if t else ""), facts=[(c.upper(), f"{res['frac'][c]:.1f}%") for c in CLASSES[:4]], evid="Full-scene K-means segmentation summary (heuristic estimate).")
    fr = res["frac"][c]; fs, m = comps(res["cls"] == CLASSES.index(c), res["conf"], 999)
    if not fs: return dict(R, title=f"NO SIGNIFICANT {c.upper()}", text=f"No {c} region above 0.3% of the scene (estimated {fr:.1f}% overall).", evid="Connected-component filter removed all candidates.")
    conf = float(np.mean([f["conf"] for f in fs[:8]])); mask = m
    ev = f"{len(fs)} major {c} region{'s' if len(fs) > 1 else ''} detected · largest = {fs[0]['area']:.1f}% of scene · heuristic K-means estimate"
    if any(w in ql for w in ("how much", "percent", "%")): title, text = f"{c.upper()} ≈ {fr:.1f}% (ESTIMATED)", f"About {fr:.1f}% of analysed pixels are classified as {c}."
    elif "how many" in ql: title, text = f"{len(fs)} {c.upper()} REGION{'S' if len(fs) > 1 else ''}", f"{len(fs)} connected regions above 0.3% of the scene."
    elif "largest" in ql:
        k, l, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8)); mask = l == 1 + int(np.argmax(st[1:, 4]))
        title, text = f"LARGEST {c.upper()} REGION: {fs[0]['where']}", f"{fs[0]['area']:.1f}% of the scene, centroid {fs[0]['centroid']}, bbox {fs[0]['box']}."
    elif "significant" in ql or "is there" in ql:
        title = f"{c.upper()}: {'SIGNIFICANT' if fr >= 15 else 'LIMITED'} ({fr:.1f}%)"; text = f"Estimated {fr:.1f}% of the scene ({'above' if fr >= 15 else 'below'} the 15% significance threshold used here)."
    elif dom: title, text = f"DOMINANT COVER: {c.upper()} {fr:.0f}%", f"{c.title()} is the largest estimated class; main concentration in the {_where(fs)}."
    else: title, text = f"{len(fs)} {c.upper()} REGION{'S' if len(fs) > 1 else ''} DETECTED", f"{c.title()} is concentrated in the {_where(fs)} of the scene. Largest region: {fs[0]['where']} ({fs[0]['area']:.1f}%)."
    return dict(title=title, text=text, mask=mask, heat=_heat(m), conf=conf, evid=ev, facts=[("REGIONS", len(fs)), ("LARGEST", f"{fs[0]['area']:.1f}% of scene"), ("LOCATION", _where(fs).replace(" and ", " / ")), ("SHARE", f"{fr:.1f}% of scene")])

def seg_overlay(res, a=0.5):
    o = np.zeros_like(res["img"])
    for i, c in enumerate(CLASSES): o[res["cls"] == i] = RGB[c]
    return cv2.addWeighted(res["img"], 1 - a, o, a, 0)

def class_view(res, c):
    m = res["cls"] == CLASSES.index(c); g = cv2.cvtColor(cv2.cvtColor(res["img"], cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    o = (g * 0.3).astype(np.uint8); o[m] = (0.5 * res["img"][m] + 0.5 * np.array(RGB[c])).astype(np.uint8); return o

def heat_overlay(img, h):
    hc = cv2.cvtColor(cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    w = (h[..., None] * 0.75).astype(np.float32); return (img * (1 - w) + hc * w).astype(np.uint8)

def highlight(img, mask):
    o = img.copy(); o[mask] = (0.45 * o[mask] + 0.55 * np.array([34, 211, 238])).astype(np.uint8)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cs, -1, (255, 255, 255), 1); return o

def feat_overlay(img, feats, sel=None):
    o = img.copy()
    for f in feats:
        x, y, w, h = f["box"]; col = RGB[f["cls"]]; t = 3 if f["name"] == sel else 1
        cv2.rectangle(o, (x, y), (x + w, y + h), col, t); cv2.putText(o, f["name"].split()[-1], (x + 3, y + 14), cv2.FONT_HERSHEY_PLAIN, 1, col, 1)
    return o

def change(a, b):
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA); aligned = False
    try:
        ga, gb = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (a, b)]; M = np.eye(2, 3, dtype=np.float32)
        _, M = cv2.findTransformECC(ga, gb, M, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        b = cv2.warpAffine(b, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE); aligned = True
    except cv2.error: pass
    la, lb = [cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2LAB), (0, 0), 2).astype(np.float32) for x in (a, b)]
    d = np.linalg.norm(la - lb, axis=2); d8 = (np.clip(d / (d.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    thr = max(20.0, float(cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]) * float(d.max()) / 255)   # Otsu, with a LAB noise floor of 20
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    H, W = m.shape; k, l, st, cen = cv2.connectedComponentsWithStats(m); ca, cb = segment(a)["cls"], segment(b)["cls"]; keep = np.zeros((H, W), bool); regs = []
    for i in range(1, k):   # drop noise blobs (<0.2% of image), label each remaining region
        if st[i, 4] < 0.002 * H * W: continue
        rm = l == i; keep |= rm; v, cnt = np.unique(ca[rm] * 5 + cb[rm], return_counts=True); j = cnt.argmax(); pa, pb = divmod(int(v[j]), 5)
        lab = f"{CLASSES[pa]} → {CLASSES[pb]}" if pa != pb and cnt[j] / rm.sum() >= 0.5 else "VISUAL CHANGE DETECTED — SEMANTIC TRANSITION UNCERTAIN"
        regs.append(dict(where=region(cen[i], W, H), area=float(st[i, 4] / (H * W) * 100), label=lab))
    regs.sort(key=lambda r: -r["area"]); m = keep
    def p(x): return float((x & m).mean() * 100)
    T = lambda s_, t_: (ca == CLASSES.index(s_)) & (cb == CLASSES.index(t_))
    items = [("vegetation → built-up", p(T("vegetation", "built-up"))), ("vegetation → bare land", p(T("vegetation", "bare land"))),
             ("water → land", p((ca == 0) & (cb != 0))), ("land → water", p((ca != 0) & (cb == 0))), ("built-up expansion (any class)", p((cb == 2) & (ca != 2)))]
    stt = lambda v: "Detected" if v >= 1 else "Possible" if v >= 0.3 else "Not detected"
    dn = (np.clip(d / (d.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    dmap = cv2.cvtColor(cv2.applyColorMap(dn, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    ov = b.copy(); cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([249, 115, 22])).astype(np.uint8); cv2.drawContours(ov, cs, -1, (255, 140, 40), 2)
    pct = float(m.mean() * 100)
    return dict(mean_diff=float(d[m].mean()) if m.any() else 0.0, largest=regs[0] if regs else None, a=a, b=b, diff=dmap, overlay=ov, pct=pct, regions=len(regs), regs=regs, aligned=aligned, items=[(n, v, stt(v)) for n, v in items],
                warn="Very large global difference: likely illumination/season/registration effects. Interpret with caution." if pct > 35 else "")
