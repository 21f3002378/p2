# orchestrator.py – main quiz flow controller

import json
import httpx
from datetime import datetime, timedelta
from urllib.parse import urlparse
from engine.browser import open_page, extract
from engine.llm import get_structured_response
from engine.code_runner import run_python
import re
import asyncio


def clean_url(u: str) -> str:
    if not u:
        return None
    return re.sub(r"<[^>]*>", "", u).strip()


async def run_quiz_flow(payload: dict):
    print("3 Starting quiz flow")
    email = payload.get("email")
    secret = payload.get("secret")
    current_url = payload.get("url")
    q_num = 0

    deadline = datetime.utcnow() + timedelta(seconds=3600)
    history = []
    submit_url = "https://tds-llm-analysis.s-anand.net/submit"
    while datetime.utcnow() < deadline and current_url:
        q_num += 1

        # --- Load page ---
        p, browser, ctx, page = await open_page(current_url)
        data = await extract(page)
        page_text = data #.get("text") or data.get("html")
        print(f'6 ---{q_num} Page loaded ---')
        # --- LLM extraction ---
        llm_json = await get_structured_response(
            page_text, #[:6000],
            "Extract answer and next_url",
            email=email,
            secret=secret,
            submit_url=submit_url
        )

        #answer = llm_json.get("answer")
        python_code = llm_json.get("python_code")
        #next_url = llm_json.get("next_url")
        submit_url = llm_json.get("submit_url")

        # fallback if no answer
        #if answer is None and not python_code:
            #answer = urlparse(current_url).path
        answer = "No answer"
        # --- Execute python code if needed ---
        if python_code:
            print("9 Executing python code to get answer")
            answer = run_python(python_code)

        # --- Submit answer ---
        payload_submit = {
            "email": email,
            "secret": secret,
            "url": current_url,
            "answer": answer
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(submit_url, json=payload_submit)
            resp.raise_for_status()
            resp_json = resp.json()
        print(f"10 Submitted answer for Q{q_num}: {answer} (correct: {resp_json})")
        
        history.append({
            "q": q_num,
            "url": current_url,
            "answer": answer,
            "correct": resp_json.get("correct")
        })

        # --- Next URL ---
        if resp_json.get("url"):
            current_url = resp_json.get("url")
        # elif resp_json['correct'] == False:
        #     current_url = next_url
        # else:
        #     current_url = None  # quiz finished

        await ctx.close()
        await browser.close()
        await p.stop()

    return history

