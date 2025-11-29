import os
import sys
import asyncio
import json
import re
import math
import traceback
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

# Windows async fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request, HTTPException
import httpx
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

# Configuration
EXPECTED_SECRET = "elephant"
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY", "")
AIPIPE_API_URL = os.getenv("AIPIPE_API_URL", "https://aipipe.org/openrouter/v1/chat/completions")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-5-nano")

TOTAL_TIME_LIMIT = 2 * 60
HTTPX_TIMEOUT = 30.0

app = FastAPI(title="LLM Analysis Quiz Orchestrator")
SCRAPED_CACHE: Dict[str, str] = {}

# ============ SYSTEM PROMPTS ============

def get_code_generation_prompt(file_type: str) -> str:
    """Get system prompt for code generation"""
    csv_prompt = """You are an expert Python data analyst. Write Python code to analyze CSV data CORRECTLY.

KEY RULES:
1. Read page context - it tells you WHAT to calculate
2. If page says "Cutoff: X" - use X as threshold, NOT as answer
3. Common tasks:
   - "count above X" → count values > X
   - "count below X" → count values < X
   - "sum all" → sum all values
   - "percentage" → (count > cutoff / total) * 100
4. NEVER use the cutoff value as the final answer
5. Print f"ANSWER:{answer}" where answer is your calculation

Use: pandas, numpy, io, re, math, scipy, sklearn"""

    pdf_prompt = """Write Python code to analyze PDF text.
Use regex to extract data. End with print(f"ANSWER:{answer}")"""

    text_prompt = """Write Python code to analyze text.
Extract patterns or values. End with print(f"ANSWER:{answer}")"""
    
    if file_type.lower() == "csv":
        return csv_prompt
    elif file_type.lower() == "pdf":
        return pdf_prompt
    else:
        return text_prompt

# ============ CSV INSPECTION ============

def inspect_csv(csv_content: str) -> Dict[str, Any]:
    """Inspect CSV structure"""
    lines = csv_content.strip().split('\n')
    if not lines:
        return {"error": "Empty CSV"}
    
    headers = lines[0].split(',')
    sample_rows = [lines[i].split(',') for i in range(1, min(6, len(lines)))]
    
    return {
        "headers": headers,
        "num_columns": len(headers),
        "sample_rows": sample_rows,
        "total_rows": len(lines) - 1
    }

# ============ FALLBACK ANALYSIS ============

def fallback_csv_analysis(csv_content: str, page_text: str = "") -> Any:
    """Fallback CSV analysis"""
    print(f"[FALLBACK] Running fallback analysis...")
    
    try:
        import pandas as pd
        import io
        import numpy as np
        
        df = pd.read_csv(io.StringIO(csv_content))
        print(f"[FALLBACK] Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        # Look for cutoff
        cutoff_match = re.search(r'[Cc]utoff[:\s]+(\d+)', page_text)
        cutoff = int(cutoff_match.group(1)) if cutoff_match else None
        
        if df.shape[1] == 1:
            col_name = df.columns[0]
            values = pd.to_numeric(df[col_name], errors='coerce')
            
            if cutoff is not None:
                count_above = int((values > cutoff).sum())
                count_below = int((values < cutoff).sum())
                pct_above = int((count_above / len(values)) * 100) if len(values) > 0 else 0
                
                print(f"[FALLBACK] Cutoff: {cutoff}")
                print(f"[FALLBACK] Count above: {count_above}, below: {count_below}, %: {pct_above}")
                
                if 'above' in page_text.lower():
                    return count_above
                elif 'below' in page_text.lower():
                    return count_below
                elif 'percent' in page_text.lower():
                    return pct_above
                else:
                    return count_above
            
            total = int(values.sum())
            print(f"[FALLBACK] Sum: {total}")
            return total
        
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            total = int(numeric_df.sum().sum())
            return total
            
    except Exception as e:
        print(f"[FALLBACK] Error: {e}")
        traceback.print_exc()
    
    return None

# ============ CODE EXECUTION ============

def execute_analysis_code(code: str, data_variables: Dict[str, str]) -> Any:
    """Execute Python code safely"""
    print(f"[EXECUTOR] Executing code ({len(code)} chars)...")
    
    try:
        exec_globals = {
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            're': __import__('re'),
            'io': __import__('io'),
            'math': __import__('math'),
            **data_variables
        }
        
        import io as io_module
        output_buffer = io_module.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output_buffer
        
        try:
            exec(code, exec_globals)
        finally:
            sys.stdout = old_stdout
        
        output = output_buffer.getvalue()
        print(f"[EXECUTOR] Output:\n{output[:500]}")
        
        if "ANSWER:" in output:
            answer_str = output.split("ANSWER:")[-1].strip().split('\n')[0].strip()
            try:
                answer = int(answer_str) if '.' not in answer_str else float(answer_str)
                print(f"[EXECUTOR] Extracted answer: {answer}")
                return answer
            except ValueError:
                return answer_str
        
        return None
        
    except Exception as e:
        print(f"[EXECUTOR] Error: {e}")
        traceback.print_exc()
        return None

# ============ LLM API ============

async def fetch_aipipe_chat(messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None, retry_count: int = 0) -> Dict[str, Any]:
    """Call LLM API with retry"""
    if not AIPIPE_API_KEY:
        raise RuntimeError("AIPIPE_API_KEY not set")
    
    max_retries = 3
    payload = {
        "model": AIPIPE_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4000
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    headers = {"Authorization": f"Bearer {AIPIPE_API_KEY}", "Content-Type": "application/json"}
    
    print(f"[API] Sending request (Messages: {len(messages)})...")
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, verify=False) as client:
            resp = await client.post(AIPIPE_API_URL, json=payload, headers=headers)
            print(f"[API] Status: {resp.status_code}")
            
            if resp.status_code != 200:
                if resp.status_code == 400 and retry_count < max_retries:
                    print(f"[API] Retrying... ({retry_count + 1}/{max_retries})")
                    await asyncio.sleep(2 ** retry_count)
                    return await fetch_aipipe_chat(messages, tools, retry_count + 1)
                print(f"[API] Error: {resp.text[:300]}")
            
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"[API] Failed: {e}")
        raise

# ============ CODE GENERATION ============

async def generate_analysis_code(messages: List[Dict[str, Any]], file_type: str, content_preview: str, page_text: str = "") -> str:
    """Generate code using LLM"""
    system_prompt = get_code_generation_prompt(file_type)
    
    code_request_message = f"""Analyze this {file_type.upper()} task:

PAGE CONTEXT (tells you WHAT to calculate):
{page_text[:800]}

Data preview:
{content_preview[:800]}

Generate Python code that:
1. Reads the data
2. Performs the calculation the page asks for
3. Prints f"ANSWER:{{answer}}"

Only output valid Python code, no explanation."""
    
    messages_for_code = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": code_request_message}]
    
    print(f"[CODE_GEN] Generating {file_type.upper()} code...")
    response = await fetch_aipipe_chat(messages_for_code)
    code = response["choices"][0]["message"]["content"]
    
    # Extract from markdown
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    print(f"[CODE_GEN] Generated {len(code)} chars")
    return code

# ============ TOOL HANDLER ============

async def handle_code_generation_tool(file_url: str, file_type: str, task_desc: str, page_text: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Handle code generation and execution"""
    print(f"\n[TOOL] Starting code generation tool...")
    print(f"[TOOL] File type: {file_type}, Task: {task_desc[:50]}")
    
    content = SCRAPED_CACHE.get(file_url)
    if not content:
        for key in SCRAPED_CACHE.keys():
            if file_type.lower() in key.lower():
                content = SCRAPED_CACHE[key]
                break
    
    if not content:
        print(f"[TOOL] Content not found")
        return {"status": "error", "error": "Content not found"}
    
    print(f"[TOOL] Content loaded: {len(content)} chars")
    
    # TRY FALLBACK FIRST for CSV - it's more reliable than LLM code gen right now
    if file_type.lower() == "csv":
        print(f"[TOOL] CSV file detected - trying fallback first...")
        try:
            schema = inspect_csv(content)
            print(f"[TOOL] CSV Schema: {schema}")
            
            fallback_answer = fallback_csv_analysis(content, page_text)
            if fallback_answer is not None:
                print(f"[TOOL] Fallback succeeded with answer: {fallback_answer}")
                return {
                    "status": "ok",
                    "answer": fallback_answer,
                    "method": "fallback_first",
                    "reasoning": f"CSV analysis: {schema['total_rows']} rows, cutoff-based calculation"
                }
        except Exception as e:
            print(f"[TOOL] Fallback attempt failed: {e}")
    
    # If not CSV or fallback failed, try code generation
    try:
        code = await generate_analysis_code(messages, file_type, content, page_text)
        
        data_vars = {
            'csv_content': content if file_type.lower() == 'csv' else '',
            'pdf_content': content if file_type.lower() == 'pdf' else '',
            'content': content,
        }
        
        answer = execute_analysis_code(code, data_vars)
        
        if answer is not None:
            print(f"[TOOL] Code execution succeeded: {answer}")
            return {"status": "ok", "answer": answer, "method": "code"}
        
        print(f"[TOOL] Code execution returned None, falling back...")
        
    except Exception as e:
        print(f"[TOOL] Code generation failed: {e}")
    
    # Final fallback for any file type
    if file_type.lower() == "csv":
        try:
            fallback_answer = fallback_csv_analysis(content, page_text)
            if fallback_answer is not None:
                print(f"[TOOL] Final fallback succeeded: {fallback_answer}")
                return {"status": "ok", "answer": fallback_answer, "method": "fallback_final"}
        except Exception as e:
            print(f"[TOOL] Final fallback also failed: {e}")
    
    return {"status": "error", "error": "All methods failed"}

# ============ TOOL DEFINITIONS ============

def get_tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "generate_code_and_execute",
                "description": "Generate and execute Python code to analyze data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_url": {"type": "string"},
                        "file_type": {"type": "string", "enum": ["csv", "pdf", "text", "markdown"]},
                        "task_description": {"type": "string"}
                    },
                    "required": ["file_url", "file_type", "task_description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "process_text_data",
                "description": "Read and process text files",
                "parameters": {
                    "type": "object",
                    "properties": {"file_url": {"type": "string"}},
                    "required": ["file_url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_javascript_page",
                "description": "Fetch JavaScript-rendered page",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Submit final answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": ["number", "string"]},
                        "submit_url": {"type": "string"},
                        "secret": {"type": "string"}
                    },
                    "required": ["answer", "submit_url"]
                }
            }
        }
    ]

def process_text_data(file_url: str) -> Dict[str, Any]:
    content = SCRAPED_CACHE.get(file_url)
    if not content:
        return {"status": "error", "error": "File not found"}
    return {"status": "ok", "content_preview": content[:3000], "total_length": len(content)}

def fetch_javascript_page(url: str) -> Dict[str, Any]:
    cached = SCRAPED_CACHE.get(url)
    if cached:
        return {"status": "ok", "content": cached[:3000], "total_length": len(cached)}
    return {"status": "error", "error": f"URL not found"}

# ============ ORCHESTRATOR ============

async def analyze_page_and_solve(page, page_text: str) -> Dict[str, Any]:
    """Analyze page and solve quiz"""
    SCRAPED_CACHE.clear()
    current_url = page.url
    print(f"\n[ORCHESTRATOR] Analyzing: {current_url}")
    
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
        page_text = await page.inner_text("body")
        anchors = await page.eval_on_selector_all("a", "els => els.map(a => ({href: a.href, text: a.innerText}))")
    except Exception as e:
        print(f"[ORCHESTRATOR] Error: {e}")
        page_text = ""
        anchors = []

    SCRAPED_CACHE[current_url] = page_text
    print(f"[ORCHESTRATOR] Page text: {len(page_text)} chars, Links: {len(anchors)}")

    for a in anchors[:6]:
        href = a['href']
        if not href or href.startswith('javascript'):
            continue
        try:
            await page.goto(href, wait_until="networkidle", timeout=15000)
            content = await page.inner_text("body")
            SCRAPED_CACHE[href] = content
            print(f"[ORCHESTRATOR] Cached: {href[:60]}... ({len(content)} bytes)")
            await page.goto(current_url, wait_until="networkidle", timeout=15000)
        except Exception as e:
            print(f"[ORCHESTRATOR] Failed {href}: {str(e)[:60]}")
            try:
                async with httpx.AsyncClient(timeout=10, verify=False) as client:
                    resp = await client.get(href)
                    SCRAPED_CACHE[href] = resp.text
            except:
                pass

    system_prompt = """You are a quiz solver. Use tools to analyze pages and extract answers.
Tools: generate_code_and_execute, process_text_data, fetch_javascript_page, final_answer"""

    initial_user_message = f"""Page: {page_text[:2000]}

Files: {json.dumps({url: f'{len(content)} bytes' for url, content in SCRAPED_CACHE.items()}, indent=2)}

Solve the quiz. Use tools."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_message}
    ]

    for iteration in range(10):
        print(f"[LLM] Turn {iteration + 1}...")
        try:
            response = await fetch_aipipe_chat(messages, tools=get_tool_definitions())
        except Exception as e:
            print(f"[LLM] Failed: {e}")
            break

        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        messages.append(message)

        if tool_calls:
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                
                print(f"[LLM] Tool: {fn_name}")
                
                if fn_name == "generate_code_and_execute":
                    res = await handle_code_generation_tool(
                        fn_args.get("file_url"), fn_args.get("file_type"),
                        fn_args.get("task_description"), page_text, messages
                    )
                elif fn_name == "process_text_data":
                    res = process_text_data(fn_args.get("file_url"))
                elif fn_name == "fetch_javascript_page":
                    res = fetch_javascript_page(fn_args.get("url"))
                elif fn_name == "final_answer":
                    ans = fn_args.get("answer")
                    submit = fn_args.get("submit_url")
                    if ans and submit:
                        print(f"[LLM] Answer: {ans}")
                        return {"submit_url": submit, "answer_payload": {"answer": ans, "secret": fn_args.get("secret")}}
                    res = {"status": "error"}
                else:
                    res = {"status": "error"}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn_name,
                    "content": json.dumps(res)
                })
        else:
            messages.append({"role": "user", "content": "Use a tool to solve this."})

    return {"error": "LLM failed"}

async def run_full_quiz_flow(initial_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run full quiz flow"""
    deadline = datetime.utcnow() + timedelta(seconds=TOTAL_TIME_LIMIT)
    current_url = initial_payload.get("url")
    history = []
    
    print(f"[FLOW] Starting: {current_url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        while datetime.utcnow() < deadline and current_url:
            print(f"\n[FLOW] Navigating: {current_url}")
            try:
                await page.goto(current_url, wait_until="networkidle", timeout=20000)
            except Exception as e:
                print(f"[FLOW] Nav error: {e}")
                break

            result = await analyze_page_and_solve(page, "")
            
            if "error" in result:
                print(f"[FLOW] Analysis failed: {result['error']}")
                return {"status": "failed", "history": history}

            submit_url = result.get("submit_url")
            ans_payload = result.get("answer_payload", {})
            
            final_payload = {
                "email": initial_payload.get("email"),
                "secret": ans_payload.get("secret") or initial_payload.get("secret"),
                "url": current_url,
                "answer": ans_payload.get("answer")
            }
            
            print(f"[FLOW] Submitting: {submit_url}")
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(submit_url, json=final_payload)
                    data = resp.json() if resp.status_code == 200 else {"correct": False}
                    history.append({"url": current_url, "response": data})
                    
                    if data.get("correct"):
                        print("[FLOW] Correct!")
                        current_url = data.get("url")
                    else:
                        print(f"[FLOW] Wrong: {data.get('reason')}")
                        current_url = data.get("url")
                        if not current_url:
                            break

            except Exception as e:
                print(f"[FLOW] Submission error: {e}")
                break
                
            if not current_url:
                return {"status": "done", "history": history}

    return {"status": "timeout", "history": history}

# ============ ENDPOINTS ============

@app.post("/test")
async def test_endpoint(request: Request):
    """Test all functions"""
    print(f"\n[TEST] Testing system...")
    try:
        # Test 1: CSV parsing
        test_csv = "value\n100\n200\n300"
        schema = inspect_csv(test_csv)
        print(f"[TEST] ✅ CSV parsing: {schema['total_rows']} rows")
        
        # Test 2: Fallback
        result = fallback_csv_analysis(test_csv, "Cutoff: 150")
        print(f"[TEST] ✅ Fallback: {result}")
        
        # Test 3: Code execution
        code = "answer = 42\nprint(f'ANSWER:{answer}')"
        result = execute_analysis_code(code, {})
        print(f"[TEST] ✅ Code execution: {result}")
        
        # Test 4: LLM
        msg = [{"role": "user", "content": "Say OK"}]
        resp = await fetch_aipipe_chat(msg)
        print(f"[TEST] ✅ LLM API works")
        
        return {"status": "success", "tests": ["csv_parsing", "fallback", "code_execution", "llm_api"]}
    except Exception as e:
        print(f"[TEST] ❌ Failed: {e}")
        return {"status": "failed", "error": str(e)}

@app.post("/quiz")
async def quiz_endpoint(request: Request):
    """Main quiz endpoint"""
    print(f"\n[ENDPOINT] New Request")
    try:
        payload = await request.json()
        if payload.get("secret") != EXPECTED_SECRET:
            raise HTTPException(status_code=403, detail="Invalid Secret")
        
        result = await run_full_quiz_flow(payload)
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"[ENDPOINT] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)