import sys, cv2, numpy as np
sys.path.insert(0, "/home/claude/opencv-ai-hack")
from boilguard.sim import BoilSim, random_params
from boilguard.vision.surface import SurfaceMeter
p = random_params(7, overflow=True, width=960, height=720); p.liquid = "milk"; p.steam_max = 0.35
sim = BoilSim(p); calib = [sim.step() for _ in range(15)]
m = SurfaceMeter(p.cam_height_mm).calibrate(calib)
f = None
while sim.s.t < 30.0: f = sim.step()
me = m.measure(f, sim.s.t)
H = 520
def fit(img, w=None):
    h0, w0 = img.shape[:2]; w = w or int(w0 * H / h0); return cv2.resize(img, (w, H), interpolation=cv2.INTER_AREA)
# 1 raw
raw = f.copy()
# 2 edges + ellipse
g = cv2.cvtColor(np.median(np.stack(calib), 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
g = cv2.GaussianBlur(cv2.createCLAHE(2.0, (8, 8)).apply(g), (5, 5), 1.2)
e = cv2.Canny(g, 40, 110); edges = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR) // 2
cv2.ellipse(edges, m.rim.as_cv_ellipse(), (0, 220, 255), 3, cv2.LINE_AA)
# 3 polar unwrap (rows = angle) + detected front, rotated so radius runs upward
pol = cv2.remap(f, m.map_x, m.map_y, cv2.INTER_LINEAR)
for r in range(m.N_T):
    c = int(m.last_front[r]); cv2.circle(pol, (c, r), 1, (77, 122, 255), -1)
cv2.line(pol, (m.i_rim, 0), (m.i_rim, m.N_T), (255, 220, 0), 1)
pol = cv2.rotate(pol, cv2.ROTATE_90_COUNTERCLOCKWISE)
pol = cv2.resize(pol, (540, 300), interpolation=cv2.INTER_LINEAR)
# 4 annotated
ann = f.copy(); cv2.ellipse(ann, m.rim.as_cv_ellipse(), (255, 220, 0), 2, cv2.LINE_AA)
for q in m.front_points().astype(int)[::2]: cv2.circle(ann, tuple(q), 2, (77, 122, 255), -1, cv2.LINE_AA)
cv2.putText(ann, f"{me.freeboard_mm:.1f} mm below rim", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
R = int(max(m.rim.a, m.rim.b) * 1.18); cx, cy = int(m.rim.cx), int(m.rim.cy)
def crop(im):
    pad = cv2.copyMakeBorder(im, R, R, R, R, cv2.BORDER_REFLECT)
    return pad[cy:cy + 2 * R, cx:cx + 2 * R]
for name, im in (("p1.png", crop(raw)), ("p2.png", crop(edges)), ("p3.png", pol), ("p4.png", crop(ann))):
    cv2.imwrite(name, im)
print("freeboard", me.freeboard_mm, "true", sim.freeboard_mm)
