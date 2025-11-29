#!/usr/bin/env python3
"""
Debug script to see what the demo endpoint returns with full JS execution
"""

import asyncio
import json
from bs4 import BeautifulSoup
import base64
from playwright.async_api import async_playwright

DEMO_URL = "https://tds-llm-analysis.s-anand.net/demo"

async def fetch_with_playwright():
    """Fetch and render with JavaScript execution"""
    print("Fetching demo endpoint with Playwright (JS enabled)...")
    
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(DEMO_URL, wait_until='networkidle')
        
        # Wait a bit for any dynamic content
        await page.wait_for_timeout(2000)
        
        # Get the rendered content
        html = await page.content()
        
        await browser.close()
    
    return html

async def analyze_html(html):
    print("\n" + "="*60)
    print("RENDERED HTML (after JS execution):")
    print("="*60)
    print(html[:2000])
    print("\n" + "="*60)
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Get all text
    text = soup.get_text()
    print("\nPAGE TEXT:")
    print("="*60)
    print(text)
    
    # Look for JSON
    import re
    json_matches = re.findall(r'\{[^}]+\}', text)
    if json_matches:
        print("\n\nFound JSON objects:")
        for match in json_matches:
            print(match)
    
    # Look for URLs
    url_matches = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
    if url_matches:
        print("\n\nFound URLs:")
        for url in set(url_matches):
            print(f"  - {url}")
    
    # Look for any data attributes
    elements_with_data = soup.find_all(attrs={'data-': True})
    if elements_with_data:
        print(f"\n\nElements with data attributes: {len(elements_with_data)}")
        for elem in elements_with_data[:5]:
            print(elem)
    
    # Look for result divs
    result_divs = soup.find_all('div', id='result')
    if result_divs:
        print(f"\n\nFound {len(result_divs)} result divs:")
        for div in result_divs:
            print(div.get_text()[:500])

async def main():
    try:
        html = await fetch_with_playwright()
        await analyze_html(html)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())