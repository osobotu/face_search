import os
import asyncio
from playwright.async_api import async_playwright

DOWNLOAD_DIR = "data/flickr_downloads"

async def scroll_and_load_page(page):
    """Scroll to bottom of current page to load all images"""
    print("[INFO] Scrolling to load all images on current page...")
    max_scrolls = 50  # Safety limit
    scroll_attempts = 0
    consecutive_bottom_checks = 0
    
    while scroll_attempts < max_scrolls:
        # Count current images
        current_count = await page.evaluate("document.querySelectorAll('.photo-card-content').length")
        
        # Check if we're at the bottom
        at_bottom = await page.evaluate("""
            () => {
                const scrollHeight = document.documentElement.scrollHeight;
                const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
                const clientHeight = document.documentElement.clientHeight;
                return scrollTop + clientHeight >= scrollHeight - 100;
            }
        """)
        
        if at_bottom:
            consecutive_bottom_checks += 1
            print(f"[INFO] At bottom (check {consecutive_bottom_checks}/3), current image count: {current_count}")
            
            # Wait a bit longer and check again to ensure no more content loads
            if consecutive_bottom_checks >= 3:
                print(f"[INFO] Confirmed at bottom with {current_count} images after {scroll_attempts} scrolls")
                break
                
            await asyncio.sleep(3)  # Wait longer when at bottom
        else:
            consecutive_bottom_checks = 0
            print(f"[INFO] Scrolling... Current image count: {current_count}")
            await asyncio.sleep(2)
        
        # Scroll down
        await page.mouse.wheel(0, 3000)
        scroll_attempts += 1

async def get_page_images(page):
    """Extract image data from current page"""
    photo_data = await page.evaluate("""
        () => {
            const photoCards = document.querySelectorAll('.photo-card-content');
            return Array.from(photoCards).map((card, index) => {
                const img = card.querySelector('img');
                const link = card.querySelector('a.photo-link');
                return {
                    src: img ? img.src : null,
                    title: link ? link.title : `image_${index + 1}`,
                    href: link ? link.href : null
                };
            });
        }
    """)
    return photo_data

async def get_all_page_urls(page, base_url):
    """Get URLs for all pages in the album"""
    print("[INFO] Discovering all pages in the album...")
    
    try:
        await page.goto(base_url, timeout=60000)
        await page.wait_for_selector(".pagination-view", timeout=30000)
        
        # Get all page URLs
        page_urls = await page.evaluate("""
            () => {
                const paginationLinks = document.querySelectorAll('.pagination-view a');
                const urls = [];
                
                // Get all unique page URLs (excluding empty text links)
                Array.from(paginationLinks).forEach(link => {
                    const text = link.textContent.trim();
                    const href = link.href;
                    
                    // Only include links that have numeric text (page numbers)
                    if (text && /^\d+$/.test(text)) {
                        urls.push({
                            url: href,
                            pageNum: parseInt(text)
                        });
                    }
                });
                
                // Sort by page number and return URLs
                return urls
                    .sort((a, b) => a.pageNum - b.pageNum)
                    .map(item => item.url);
            }
        """)
        
        if len(page_urls) == 0:
            # If no pagination found, just use the base URL
            print("[INFO] No pagination found, treating as single page")
            return [base_url]
        
        print(f"[INFO] Found {len(page_urls)} pages:")
        for i, url in enumerate(page_urls):
            print(f"  Page {i + 1}: {url}")
        
        return page_urls
        
    except Exception as e:
        print(f"[WARN] Error discovering pages: {e}")
        print("[INFO] Falling back to single page")
        return [base_url]

async def download_images(page, photo_data, album_name, page_num, start_idx=0):
    """Download images from photo_data"""
    downloaded = 0
    for idx, data in enumerate(photo_data):
        try:
            if not data['src']:
                print(f"No image source found for item {start_idx + idx + 1}")
                continue

            img_url = data['src']
            img_title = data['title'] or f"image_{start_idx + idx + 1}"

            # Normalize URL
            if img_url.startswith("//"):
                img_url = "https:" + img_url

            # Clean filename
            base_name = (
                img_title.strip()
                .replace(" ", "_")
                .replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-")
                .replace("?", "")
                .replace("*", "")
                .replace("<", "")
                .replace(">", "")
                .replace("|", "")
                .replace('"', "")
            )
            
            # Get file extension from URL
            ext = img_url.split(".")[-1].split("?")[0]
            if not ext or len(ext) > 4:
                ext = "jpg"  # fallback
            
            filename = f"{DOWNLOAD_DIR}/{album_name}/page{page_num}_{base_name}_{start_idx + idx + 1:03d}.{ext}"

            # Check if file already exists
            if os.path.exists(filename):
                print(f"Skipping existing file: {os.path.basename(filename)}")
                downloaded += 1  # Count as downloaded to maintain progress
                continue

            # Download the image directly
            print(img_url)
            response = await page.goto(img_url)
            if response and response.status == 200:
                content = await response.body()
                with open(filename, 'wb') as f:
                    f.write(content)
                print(f"Downloaded: {os.path.basename(filename)}")
                downloaded += 1
            else:
                print(f"Failed to download {img_url}: HTTP {response.status if response else 'No response'}")

        except Exception as e:
            print(f"Error downloading image {start_idx + idx + 1}: {e}")
            
            # Fallback: try to visit individual photo page
            if data.get('href'):
                try:
                    print(f"[INFO] Trying fallback method for image {start_idx + idx + 1}")
                    await page.goto(data['href'])
                    await page.wait_for_selector("img.main-photo", timeout=10000)
                    img_url = await page.eval_on_selector("img.main-photo", "el => el.src")
                    
                    if img_url.startswith("//"):
                        img_url = "https:" + img_url
                    
                    response = await page.goto(img_url)
                    if response and response.status == 200:
                        content = await response.body()
                        with open(filename, 'wb') as f:
                            f.write(content)
                        print(f"Downloaded via fallback: {os.path.basename(filename)}")
                        downloaded += 1
                except Exception as fallback_error:
                    print(f"Fallback also failed for image {start_idx + idx + 1}: {fallback_error}")
    
    return downloaded

async def scrape_album(album_url: str, album_name="default_album", max_pages=None):
    os.makedirs(f"{DOWNLOAD_DIR}/{album_name}", exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set user agent to avoid being blocked
        await page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        try:
            # Step 1: Discover all page URLs
            all_page_urls = await get_all_page_urls(page, album_url)
            
            # Apply max_pages limit if specified
            if max_pages and max_pages < len(all_page_urls):
                all_page_urls = all_page_urls[:max_pages]
                print(f"[INFO] Limited to first {max_pages} pages")
            
            total_downloaded = 0
            
            # Step 2: Process each page
            for page_idx, page_url in enumerate(all_page_urls):
                page_num = page_idx + 1
                print(f"\n[INFO] Processing page {page_num}/{len(all_page_urls)}: {page_url}")
                
                try:
                    await page.goto(page_url, timeout=60000)
                    
                    # Wait for photo cards to load
                    await page.wait_for_selector(".photo-card-content", timeout=30000)
                    print(f"[INFO] Photo cards loaded on page {page_num}")
                    
                    # Scroll to load all images on this page
                    await scroll_and_load_page(page)
                    
                    # Extract image data from current page
                    photo_data = await get_page_images(page)
                    print(f"[INFO] Found {len(photo_data)} images on page {page_num}")
                    
                    if photo_data:
                        # Download images from current page
                        downloaded = await download_images(page, photo_data, album_name, page_num, total_downloaded)
                        total_downloaded += downloaded
                        print(f"[INFO] Downloaded {downloaded} images from page {page_num}")
                    else:
                        print(f"[WARN] No images found on page {page_num}")
                    
                    # Add delay between pages to be respectful
                    if page_num < len(all_page_urls):
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    continue
            
            print(f"\n[INFO] Scraping completed! Downloaded {total_downloaded} total images from {len(all_page_urls)} pages.")
            print(f"[INFO] Check {DOWNLOAD_DIR}/{album_name}/ for all downloaded images.")
            
        except Exception as e:
            print(f"Fatal error during scraping: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    album = input("Paste Flickr album URL: ")
    name = input("Name for album folder: ") or "flickr_album"
    
    # Optional: limit pages for testing
    max_pages_input = input("Max pages to scrape (press Enter for all): ").strip()
    max_pages = int(max_pages_input) if max_pages_input.isdigit() else None
    
    asyncio.run(scrape_album(album.strip(), album_name=name.strip(), max_pages=max_pages))