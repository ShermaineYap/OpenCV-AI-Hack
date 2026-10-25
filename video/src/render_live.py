import asyncio, os, sys, subprocess, shutil
from playwright.async_api import async_playwright
HERE = os.path.dirname(os.path.abspath(__file__))
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 19.6
async def main(times, outdir):
    shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--allow-file-access-from-files"])
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        await pg.goto("file://" + os.path.join(HERE, "live.html"))
        for i, t in enumerate(times):
            await pg.evaluate(f"render({t}, {DUR})"); await pg.evaluate("ready()")
            await pg.screenshot(path=os.path.join(outdir, f"{i:05d}.jpg"), type="jpeg", quality=92)
        await b.close()
if sys.argv[1] == "stills":
    asyncio.run(main([3.0, 9.5, 12.0, 17.5], os.path.join(HERE, "lstills")))
else:
    asyncio.run(main([i / 30 for i in range(int(DUR * 30))], os.path.join(HERE, "lframes")))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", os.path.join(HERE, "lframes", "%05d.jpg"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(HERE, "seg_live.mp4")], check=True)
