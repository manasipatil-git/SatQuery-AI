"""Generates synthetic satellite-style demo scenes (no internet needed). Drop real PNG/JPGs into data/demo to add more."""
import numpy as np, cv2, os
R = np.random.RandomState(7); H, W = 480, 640; out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
def fields(c, x0, y0, x1, y1):
    pal = [(95, 135, 60), (110, 150, 70), (80, 120, 55), (165, 145, 105), (125, 160, 85)]
    y = y0
    while y < y1:
        h = R.randint(40, 80); x = x0
        while x < x1:
            w = R.randint(50, 110); cv2.rectangle(c, (x, y), (min(x + w, x1), min(y + h, y1)), pal[R.randint(5)], -1); x += w
        y += h
def city(c, x0, y0, x1, y1):
    cv2.rectangle(c, (x0, y0), (x1, y1), (150, 150, 150), -1)
    for y in range(y0 + 6, y1 - 10, 26):
        for x in range(x0 + 6, x1 - 10, 26):
            if R.rand() < .85:
                w, h = R.randint(10, 19), R.randint(10, 19); v = R.randint(175, 235); t = (v, v - R.randint(0, 25), v - R.randint(10, 45))
                cv2.rectangle(c, (x + 3, y + 3), (x + w + 3, y + h + 3), (95, 95, 100), -1); cv2.rectangle(c, (x, y), (x + w, y + h), t, -1)
def water(c, pts): cv2.fillPoly(c, [np.array(pts, np.int32)], (28, 72, 108))
def finish(c, name):
    c = c.astype(np.float32); n = cv2.GaussianBlur(R.randn(H, W).astype(np.float32), (0, 0), 1.2) * 9
    c = np.clip(cv2.GaussianBlur(c, (0, 0), 0.8) + n[..., None], 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, name), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
def new(): c = np.zeros((H, W, 3), np.uint8); fields(c, 0, 0, W, H); return c
c = new(); city(c, 0, 0, 400, H); cv2.rectangle(c, (400, 0), (440, H), (200, 185, 140), -1)
water(c, [(440, 0), (W, 0), (W, H), (470, H), (450, 300), (470, 150)]); cv2.rectangle(c, (60, 300), (160, 380), (80, 125, 55), -1); finish(c, "coastal_city.png")
c = new(); cv2.polylines(c, [np.array([(0, 90), (150, 130), (300, 250), (450, 280), (W, 400)], np.int32)], False, (28, 72, 108), 34); city(c, 420, 40, 560, 140); finish(c, "farmland_river.png")
c = np.zeros((H, W, 3), np.uint8); city(c, 0, 0, W, H); cv2.ellipse(c, (430, 300), (130, 80), 20, 0, 360, (28, 72, 108), -1); cv2.rectangle(c, (40, 40), (200, 140), (80, 125, 55), -1); finish(c, "lakeside_urban.png")
def scene(late):
    global R; R = np.random.RandomState(11); c = new(); city(c, 20, 20, 180, 140); cv2.ellipse(c, (480, 330), (90 if late else 130, 60 if late else 90), 0, 0, 360, (28, 72, 108), -1)
    if late: city(c, 180, 20, 420, 260); R = np.random.RandomState(12)
    return c
finish(scene(False), "change_2023.png"); finish(scene(True), "change_2026.png")
