import asyncio, json, time
from playwright.async_api import async_playwright
URL = "https://5a25kqeayc5ki2btob2zvzuhhq0gocaz.lambda-url.us-east-1.on.aws/"
CURSOR = """
window.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.id = '__cur';
  c.innerHTML = '<svg width="28" height="28" viewBox="0 0 24 24"><path d="M4 2l15 11-6.5 1.2L16 21l-3 1.4-3.6-6.9L4 20z" fill="#fff" stroke="#000" stroke-width="1.4" stroke-linejoin="round"/></svg>';
  Object.assign(c.style, {position:'fixed', left:'0px', top:'0px', zIndex:2147483647, pointerEvents:'none', transform:'translate(-4px,-2px)'});
  document.body.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', e => {
    const r = document.createElement('div');
    Object.assign(r.style, {position:'fixed', left:(e.clientX-18)+'px', top:(e.clientY-18)+'px', width:'36px', height:'36px', borderRadius:'50%',
      border:'3px solid #ff7a4d', zIndex:2147483646, pointerEvents:'none', transition:'transform .45s ease-out, opacity .45s ease-out'});
    document.body.appendChild(r);
    requestAnimationFrame(() => { r.style.transform = 'scale(1.8)'; r.style.opacity = '0'; });
    setTimeout(() => r.remove(), 600);
  }, true);
});
"""
marks = {}
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1600, "height": 900}, color_scheme="dark",
                                  record_video_dir="rec2", record_video_size={"width": 1600, "height": 900})
        await ctx.add_init_script(CURSOR)
        pg = await ctx.new_page()
        await pg.route("**/fonts.googleapis.com/**", lambda r: r.abort())
        t0 = time.time(); mark = lambda k: marks.__setitem__(k, round(time.time() - t0, 2))
        await pg.goto(URL); await pg.wait_for_selector("#health:not(:empty)")
        await pg.mouse.move(800, 450); await pg.wait_for_timeout(1000)
        async def go(sel, click=True, steps=28):
            bb = await pg.locator(sel).first.bounding_box()
            await pg.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2, steps=steps)
            await pg.wait_for_timeout(250)
            if click: await pg.mouse.down(); await pg.wait_for_timeout(90); await pg.mouse.up()
            await pg.wait_for_timeout(450)
        mark("form")
        await go("#liquid", click=False); await pg.select_option("#liquid", "milk"); await pg.wait_for_timeout(400)
        await go("#heater", click=False); await pg.select_option("#heater", "ceramic"); await pg.wait_for_timeout(400)
        await go("#steam", click=False); await pg.select_option("#steam", "0.4"); await pg.wait_for_timeout(300)
        await go("label:has(input[name=ovf][value='1'])")
        await go("label:has(input[name=cook][value='absent'])")
        await go("#seed"); await pg.keyboard.type("7", delay=160)
        await go("label:has(#mirror)")
        await go("#runbtn", click=False); mark("click")
        await pg.mouse.down(); await pg.wait_for_timeout(90); await pg.mouse.up()
        await pg.wait_for_function("/Done|Error/.test(document.querySelector('#simstatus').textContent)", timeout=300000)
        print("status:", await pg.inner_text("#simstatus")); mark("done")
        await pg.mouse.move(1100, 300, steps=20); await pg.wait_for_timeout(1200)
        await pg.evaluate("document.querySelector('#simvideo').scrollIntoView({behavior:'smooth', block:'start'})")
        await pg.wait_for_timeout(1300)
        await pg.evaluate("(()=>{const v=document.querySelector('#simvideo'); v.currentTime=24; v.play();})()")
        await pg.wait_for_timeout(5000); mark("charts")
        await pg.evaluate("window.scrollTo({top: document.querySelector('#kpis').getBoundingClientRect().top + scrollY - 20, behavior:'smooth'})")
        await pg.mouse.move(1000, 520, steps=30); await pg.wait_for_timeout(6500)
        mark("trace")
        await pg.evaluate("window.scrollTo({top: document.querySelector('#simdetails').getBoundingClientRect().top + scrollY - 20, behavior:'smooth'})")
        await pg.wait_for_timeout(1500)
        bb = await pg.locator("#trace").bounding_box()
        await pg.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + 200, steps=25); await pg.wait_for_timeout(1200)
        for _ in range(6):
            await pg.mouse.wheel(0, 140); await pg.wait_for_timeout(1100)
        await pg.wait_for_timeout(1500); mark("end")
        marks["file"] = await pg.video.path(); await ctx.close(); await b.close()
    json.dump(marks, open("demo2_marks.json", "w"), indent=1); print(marks)
asyncio.run(main())
