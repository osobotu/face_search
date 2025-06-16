import asyncio
from playwright.async_api import async_playwright

async def test_pagination(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set to False to see what's happening
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_selector(".pagination-view", timeout=10000)
        
        # Get all pagination info
        info = await page.evaluate("""
            () => {
                const links = document.querySelectorAll('.pagination-view a');
                return Array.from(links).map(link => ({
                    href: link.href,
                    text: link.textContent.trim(),
                    rel: link.getAttribute('rel')
                }));
            }
        """)
        
        print("All pagination links:")
        for link in info:
            print(f"  {link}")
        
        await browser.close()

# Test with your URL
asyncio.run(test_pagination("https://www.flickr.com/photos/cmu-africa/albums/72177720324927574/"))