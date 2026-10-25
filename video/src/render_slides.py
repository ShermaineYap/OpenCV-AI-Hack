import asyncio, json, os, sys, shutil, subprocess
from playwright.async_api import async_playwright
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "durations.json")))
LEAD, TAIL = 0.35, 0.55
SCENES = ["team", "problem", "lag", "idea", "vision", "predict", "aws", "results", "limits", "responsible", "close"]
def dur(s): return D[s] + LEAD + TAIL + (2.5 if s == "close" else 0)
async def run(mode, only=None):
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--allow-file-access-from-files"])
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        await pg.goto("file://" + os.path.join(HERE, "slides.html")); await pg.evaluate("ready()")
        for s in (only or SCENES):
            T = dur(s)
            if mode == "stills":
                for frac in (0.5, 0.97):
                    await pg.evaluate(f"render('{s}', {T*frac}, {T})")
                    await pg.screenshot(path=os.path.join(HERE, "stills", f"{s}_{int(frac*100)}.jpg"), type="jpeg", quality=85)
                continue
            out = os.path.join(HERE, "sf", s); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
            prev_sig, prev_path, shots = None, None, 0
            n = int(round(T * 30))
            for i in range(n):
                t = i / 30 - LEAD + LEAD     # scene clock starts at 0
                sig = await pg.evaluate(f"render('{s}', {t}, {T})")
                path = os.path.join(out, f"{i:05d}.jpg")
                if sig == prev_sig: os.link(prev_path, path)
                else:
                    await pg.screenshot(path=path, type="jpeg", quality=92); shots += 1
                prev_sig, prev_path = sig, path
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", os.path.join(out, "%05d.jpg"),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                            os.path.join(HERE, f"seg_{s}.mp4")], check=True)
            print(s, f"{T:.2f}s", n, "frames,", shots, "screenshots", flush=True)
        await b.close()
if __name__ == "__main__":
    mode = sys.argv[1]; only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    os.makedirs(os.path.join(HERE, "stills"), exist_ok=True)
    asyncio.run(run(mode, only))
