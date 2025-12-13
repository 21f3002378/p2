# browser.py — playwright wrapper
from playwright.async_api import async_playwright

async def open_page(url):
    print(f"4 Opening page: {url}")
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    ctx = await browser.new_context()
    page = await ctx.new_page()
    await page.goto(url, wait_until="networkidle")
    return p, browser, ctx, page


async def extract(page):
    print("5 Extracting page content")
    html = await page.content()

    # get inner text
    try:
        text = await page.inner_text("body")
    except:
        text = html

    # extract all <a href="">
    links = await page.evaluate("""
() => {
    const urls = new Set();

    const attrs = ["href", "src", "data", "content"];

    document.querySelectorAll("*").forEach(el => {
        attrs.forEach(attr => {
            const val = el.getAttribute && el.getAttribute(attr);
            if (val) urls.add(val);
        });

        // inline CSS: background-image: url(...)
        const style = window.getComputedStyle(el).backgroundImage;
        if (style && style.startsWith("url(")) {
            urls.add(style.slice(5, -2)); // remove url("...") wrapper
        }
    });

    return Array.from(urls);
}
""")


    return {
        "html": html,
        "text": text,
        "links": links
    }

