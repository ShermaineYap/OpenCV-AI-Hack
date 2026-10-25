import asyncio, json, time
from playwright.async_api import async_playwright
URL = "https://5a25kqeayc5ki2btob2zvzuhhq0gocaz.lambda-url.us-east-1.on.aws/"
marks = {}
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1, color_scheme="dark",
                                  record_video_dir="rec", record_video_size={"width": 1600, "height": 900})
        pg = await ctx.new_page()
        await pg.route("**/fonts.googleapis.com/**", lambda r: r.abort())
        t0 = time.time(); mark = lambda k: marks.__setitem__(k, round(time.time() - t0, 2))
        await pg.goto(URL); await pg.wait_for_selector("#health:not(:empty)"); await pg.wait_for_timeout(1200)
        mark("form")
        for sel, val in (("#liquid", "milk"), ("#heater", "ceramic"), ("#steam", "0.4")):
            await pg.select_option(sel, val); await pg.wait_for_timeout(700)
        await pg.click("input[name=ovf][value='1']", force=True); await pg.wait_for_timeout(500)
        await pg.click("input[name=cook][value='absent']", force=True); await pg.wait_for_timeout(500)
        await pg.select_option("#mode", "auto_veto"); await pg.wait_for_timeout(500)
        await pg.fill("#seed", "7"); await pg.wait_for_timeout(500)
        await pg.check("#mirror"); await pg.wait_for_timeout(700)
        mark("click"); await pg.click("#runbtn")
        await pg.wait_for_function("/Done|Error/.test(document.querySelector('#simstatus').textContent)", timeout=300000)
        print("status:", await pg.inner_text("#simstatus"))
        mark("done"); await pg.wait_for_timeout(1500)
        marks["video_url"] = await pg.get_attribute("#simvideo", "src")
        # scroll to the video and play from the interesting part
        await pg.evaluate("document.querySelector('#simvideo').scrollIntoView({behavior:'smooth', block:'start'})")
        await pg.wait_for_timeout(1200)
        await pg.evaluate("(()=>{const v=document.querySelector('#simvideo'); v.currentTime=19; v.play();})()")
        await pg.wait_for_timeout(500); mark("video")
        bb = await pg.evaluate("(()=>{const r=document.querySelector('#simvideo').getBoundingClientRect();return [r.x,r.y,r.width,r.height]})()")
        marks["video_bbox"] = bb
        await pg.wait_for_timeout(19000)
        mark("charts")
        await pg.evaluate("window.scrollTo({top: document.querySelector('#kpis').getBoundingClientRect().top + scrollY - 20, behavior:'smooth'})")
        await pg.wait_for_timeout(7000)
        mark("trace")
        await pg.evaluate("window.scrollTo({top: document.querySelector('#simdetails').getBoundingClientRect().top + scrollY - 20, behavior:'smooth'})")
        await pg.wait_for_timeout(5000)
        await pg.evaluate("document.querySelector('#trace').scrollTo({top: 400, behavior:'smooth'})")
        await pg.wait_for_timeout(5000)
        await pg.evaluate("document.querySelector('#trace').scrollTo({top: 9999, behavior:'smooth'})")
        await pg.wait_for_timeout(5000); mark("end")
        vp = await pg.video.path(); await ctx.close(); await b.close()
        marks["file"] = vp
    json.dump(marks, open("demo_marks.json", "w"), indent=1); print(marks)
asyncio.run(main())
