# main.py
import os
import asyncio
import json
import re
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import sys
import asyncio

if sys.platform.startswith("win"):
    # Use the older SelectorEventLoop which supports subprocesses properly
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel, HttpUrl
import httpx
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# -------- Configuration (use env vars) ----------
EXPECTED_SECRET = "elephant"         # student's SECRET
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")                 # must be set
AIPIPE_API_URL = "https://api.aipipe.io/v1/chat/completions"
AIPIPE_MODEL = "gpt-4-turbo"

MAX_PAYLOAD_BYTES = 1 * 1024 * 1024  # 1MB
# total allowed time from receiving original POST to finish whole multi-URL flow
TOTAL_TIME_LIMIT = 3 * 60  # 3 minutes (180 seconds)
# safe internal per-network timeouts
HTTPX_TIMEOUT = 30.0

# -------- App setup ----------
app = FastAPI(title="LLM Analysis Quiz Orchestrator")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl
    # accept extra fields
    class Config:
        extra = "allow"

# -------- Utilities ----------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

async def fetch_aipipe_chat(messages: List[Dict[str, Any]],
                            functions: Optional[List[Dict[str, Any]]] = None,
                            function_call: Optional[Any] = None) -> Dict[str, Any]:
    """
    Call AIPipe chat completions endpoint (OpenAI-compatible API).
    We pass messages and (optionally) function definitions and return the JSON response.
    """
    if not AIPIPE_API_KEY:
        raise RuntimeError("AIPIPE_API_KEY is not set in environment")

    payload: Dict[str, Any] = {
        "model": AIPIPE_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1400
    }
    if functions:
        payload["functions"] = functions
    if function_call is not None:
        payload["function_call"] = function_call  # allow explicit function_call or "auto"

    headers = {"Authorization": f"Bearer {AIPIPE_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        resp = await client.post(AIPIPE_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

def build_function_definitions() -> List[Dict[str, Any]]:
    """
    Return function definitions metadata (OpenAI-style) that instruct the LLM about
    the available solver functions and their expected params.
    Fill parameters with clear types and descriptions so LLM can pick correctly.
    """
    return [
        {
            "name": "spreadsheet_solver",
            "description": "Solve tasks that require reading spreadsheets / CSVs and aggregating or computing values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_urls": {"type": "array", "items": {"type": "string"}, "description": "URLs of spreadsheet files to download (csv/xlsx)"},
                    "instructions": {"type": "string", "description": "Question-specific instruction for spreadsheet processing"}
                },
                "required": ["file_urls"]
            }
        },
        {
            "name": "pdf_solver",
            "description": "Solve tasks that require downloading and parsing PDFs (text extraction, tables, pages).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_url": {"type": "string", "description": "URL of the PDF to download"},
                    "instructions": {"type": "string", "description": "Question-specific instruction for PDF analysis"}
                },
                "required": ["file_url"]
            }
        },
        {
            "name": "textmd_solver",
            "description": "Process plaintext or markdown found on the page; perform cleansing, transformation, or summarization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to process"},
                    "instructions": {"type": "string"}
                },
                "required": ["text"]
            }
        },
        {
            "name": "vision_solver",
            "description": "Handle images: OCR, image analysis, or convert image->text or extract numeric values from images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_urls": {"type": "array", "items": {"type": "string"}},
                    "instructions": {"type": "string"}
                },
                "required": ["image_urls"]
            }
        },
        {
            "name": "other_api_solver",
            "description": "Call external APIs as instructed by the quiz (APIs that require headers or special auth).",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_url": {"type": "string"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "GET"},
                    "headers": {"type": "object"},
                    "body": {"type": "object"},
                    "instructions": {"type": "string"}
                },
                "required": ["api_url"]
            }
        },
        {
            "name": "base64_solver",
            "description": "Decode/encode base64 data URI strings to files or extract embedded payloads.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_uris": {"type": "array", "items": {"type": "string"}},
                    "instructions": {"type": "string"}
                },
                "required": ["data_uris"]
            }
        },
        {
            "name": "general_llm_solver",
            "description": "When none of the above specialized solvers fit, use the LLM to either produce an answer or produce runnable code (python) for local execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Collected page context, brief"},
                    "instructions": {"type": "string", "description": "What to do"}
                },
                "required": ["context", "instructions"]
            }
        }
    ]

# -------- Solver function stubs (you will implement internals) ----------
async def fetch_aipipe_chat(messages: List[Dict], functions: Any = None) -> str:
    import aiohttp
    data = {
        "model": "gpt-5-nano",
        "messages": messages
    }
    if functions:
        data["functions"] = functions

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=data) as resp:
            try:
                res = await resp.json()
                return res["choices"][0]["message"]["content"]
            except:
                return None


# --------------------------
# SPREADSHEET SOLVER
# --------------------------
async def spreadsheet_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download CSV/XLSX and perform requested operations.
    Example operation: sum of a column named 'value'.
    """
    file_url = params.get("file_url")
    column = params.get("column")
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        resp = await client.get(file_url)
        content = resp.content

    # Detect type
    if file_url.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
    else:
        df = pd.read_excel(BytesIO(content))

    result = None
    if column and column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            result = float(df[column].sum())
        else:
            result = df[column].astype(str).str.cat(sep=" ")

    return {"status": "ok", "type": "spreadsheet", "result": result}


# --------------------------
# PDF SOLVER
# --------------------------
async def pdf_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download PDF, extract text/tables.
    """
    pdf_url = params.get("pdf_url")
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        resp = await client.get(pdf_url)
        content = resp.content

    reader = PdfReader(BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return {"status": "ok", "type": "pdf", "text_sample": text[:1000]}


# --------------------------
# TEXT / MARKDOWN SOLVER
# --------------------------
async def textmd_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze/clean text.
    """
    text = params.get("text", "")
    # Example: word count & first 100 chars
    word_count = len(text.split())
    snippet = text[:100]
    return {"status": "ok", "type": "textmd", "word_count": word_count, "snippet": snippet}


# --------------------------
# VISION SOLVER
# --------------------------
async def vision_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download base64 images, apply OCR or other analysis.
    """
    results = []
    for b64_uri in params.get("images", []):
        m = re.match(r"data:(?P<mime>[^;]+);base64,(?P<b64>.+)", b64_uri)
        if not m:
            results.append({"error": "not a valid data URI"})
            continue

        img_bytes = base64.b64decode(m.group("b64"))
        img = Image.open(BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        results.append({"ocr_text_sample": text[:200]})

    return {"status": "ok", "type": "vision", "results": results}


# --------------------------
# OTHER API SOLVER
# --------------------------
async def other_api_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    api_url = params.get("api_url")
    method = params.get("method", "GET").upper()
    headers = params.get("headers", {})
    body = params.get("body", None)

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        resp = await client.request(method, api_url, headers=headers, json=body)
        text = resp.text
        if len(text) > 1000:
            text = text[:1000] + "...(truncated)"
        return {"status": "ok", "status_code": resp.status_code, "text_sample": text}


# --------------------------
# BASE64 SOLVER
# --------------------------
async def base64_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    results = []
    for uri in params.get("data_uris", []):
        m = re.match(r"data:(?P<mime>[^;]+);base64,(?P<b64>.+)", uri)
        if not m:
            results.append({"uri": uri, "error": "not a valid data URI"})
            continue
        b64_data = m.group("b64")
        size_bytes = len(b64_data) * 3 // 4
        results.append({"mime": m.group("mime"), "size_bytes": size_bytes})
    return {"status": "ok", "type": "base64", "results": results}


# --------------------------
# GENERAL LLM SOLVER
# --------------------------
async def general_llm_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    context = params.get("context", "")
    instructions = params.get("instructions", "")

    messages = [
        {"role": "system", "content": "You are a helpful assistant that returns either an answer JSON or python code to compute the answer."},
        {"role": "user", "content": f"Context: {context}\nInstructions: {instructions}\nRespond with JSON containing 'answer_payload' and 'submit_url' when possible."}
    ]

    resp_text = await fetch_aipipe_chat(messages)
    return {"status": "ok", "aipipe_response": resp_text}
# Map function-name to local callable
LOCAL_FUNCTION_DISPATCH = {
    "spreadsheet_solver": spreadsheet_solver,
    "pdf_solver": pdf_solver,
    "textmd_solver": textmd_solver,
    "vision_solver": vision_solver,
    "other_api_solver": other_api_solver,
    "base64_solver": base64_solver,
    "general_llm_solver": general_llm_solver
}

# -------- Main orchestration: parse page, ask LLM which function to call, dispatch, loop until done ----------
async def analyze_page_and_solve(page, original_payload: Dict[str, Any], time_deadline: datetime) -> Dict[str, Any]:
    """
    1) Collect page content, links, script sources, images, data URIs, inline tables.
    2) Build a context and ask AIPipe which function should solve it (function_call).
    3) Dispatch to the selected local function, get result.
    4) Send function result back to AIPipe to produce final answer_payload and submit_url.
    5) Return the submit_url and answer_payload to the outer loop for submission.

    Note: This function follows the OpenAI-style function-calling handshake:
      - messages with system/user/context
      - functions=build_function_definitions()
      - model returns a function_call -> we call the corresponding local function
      - we then pass the function result back to the model and request a final answer
    """
    # 1) collect page artifacts
    page_html = await page.content()
    page_text = await page.inner_text("body") if await page.query_selector("body") else ""
    anchors = await page.eval_on_selector_all("a", "els => els.map(a=>({href:a.href, text:a.textContent}))")
    images = await page.eval_on_selector_all("img", "els => els.map(i=>({src:i.src, alt:i.alt}))")
    # gather any data: URIs on page
    data_uris = re.findall(r"data:[^'\"\s<>]+base64,[A-Za-z0-9+/=]+", page_html)
    # gather script tags (external)
    scripts = await page.eval_on_selector_all("script", "els => els.map(s=>({src:s.src, text: s.textContent ? s.textContent.slice(0,200) : null}))")
    # find forms or explicit submit URLs in js variables
    forms = await page.eval_on_selector_all("form", "els => els.map(f=>({action:f.action, method:f.method}))")

    context = {
        "url": original_payload.get("url"),
        "email": original_payload.get("email"),
        "page_text_snippet": page_text[:2000],
        "anchors": anchors[:40],
        "images": images[:40],
        "forms": forms,
        "data_uris_count": len(data_uris),
        "scripts_count": len(scripts),
    }

    # 2) Ask AIPipe which function to call (functions metadata)
    functions = build_function_definitions()
    messages = [
        {"role": "system", "content": "You are an orchestration assistant. Decide which specialized solver function should handle the task."},
        {"role": "user", "content": f"Given this page context (URL: {context['url']}), the page text snippet and list of anchors/images/forms are provided. Choose the best solver function and provide parameters as JSON. Context: {json.dumps(context)[:4000]}\nAlso ensure the returned parameters include any file URLs the model expects to be downloaded."}
    ]

    # Ask model to pick a function automatically
    aipipe_resp = await fetch_aipipe_chat(messages, functions=functions, function_call="auto")
    # Parse model response
    # AIPipe returns an object similar to OpenAI. We expect:
    # aipipe_resp['choices'][0]['message'] either contains 'function_call' or 'content'
    try:
        choice = aipipe_resp["choices"][0]
        message = choice.get("message", {})
    except Exception:
        raise RuntimeError("Invalid AIPipe response format")

    func_call = message.get("function_call")
    if not func_call:
        # model didn't call a function; fallback to general_llm_solver by packaging the context
        selected_name = "general_llm_solver"
        func_args = {"context": json.dumps(context)[:4000], "instructions": "Produce an answer_payload and submit_url if possible."}
    else:
        selected_name = func_call.get("name")
        raw_args = func_call.get("arguments", "{}")
        # raw_args may be a stringified JSON
        try:
            func_args = json.loads(raw_args)
        except Exception:
            func_args = {"raw": raw_args}

    # 3) Dispatch to local function
    local_fn = LOCAL_FUNCTION_DISPATCH.get(selected_name)
    if not local_fn:
        # unknown function -> fallback
        selected_name = "general_llm_solver"
        func_args = {"context": json.dumps(context)[:4000], "instructions": "Selected function was unknown. Provide either an answer_payload+submit_url or code."}
        local_fn = general_llm_solver

    # Run the local solver (respecting time_deadline)
    time_left = (time_deadline - datetime.utcnow()).total_seconds()
    if time_left <= 3:
        raise RuntimeError("Not enough time left to solve page")
    try:
        # ensure we don't run too long
        fn_result = await asyncio.wait_for(local_fn(func_args), timeout=min(30.0, time_left - 2))
    except asyncio.TimeoutError:
        fn_result = {"status": "error", "error": "local solver timed out", "function": selected_name}
    except Exception as e:
        fn_result = {"status": "error", "error": str(e), "function": selected_name}

    # 4) Give function result back to AIPipe to produce final answer_payload & submit_url
    followup_messages = [
        {"role": "system", "content": "You are a finalizing assistant. Given the page context and the function result, produce the final JSON payload to post to the quiz submit endpoint."},
        {"role": "user", "content": f"Page context: {json.dumps(context)[:4000]}\nFunction called: {selected_name}\nFunction result: {json.dumps(fn_result)[:8000]}\nProvide a JSON object with keys: 'submit_url' and 'answer_payload'. 'answer_payload' must include email and secret from the original request."}
    ]
    # ask model for final answer (no functions this time)
    final_resp = await fetch_aipipe_chat(followup_messages, functions=None)
    # parse final_resp to find content or JSON
    final_choice = final_resp.get("choices", [])[0]
    final_message = final_choice.get("message", {})
    final_text = final_message.get("content") or final_message.get("delta", {}).get("content", "")
    # try to find JSON in final_text
    submit_url = None
    answer_payload = None
    try:
        candidate = re.search(r"\{.*\}", final_text, flags=re.S)
        if candidate:
            j = json.loads(candidate.group(0))
            submit_url = j.get("submit_url")
            answer_payload = j.get("answer_payload")
    except Exception:
        submit_url = None
        answer_payload = None

    # If LLM didn't return structured JSON, attempt to parse common patterns (fallback)
    # Also ensure we inject email and secret in payload
    if isinstance(answer_payload, dict):
        answer_payload.setdefault("email", original_payload.get("email"))
        answer_payload.setdefault("secret", original_payload.get("secret"))
        answer_payload.setdefault("url", original_payload.get("url"))
    else:
        # Fallback: if final_text contains "submit_url: ..." and "answer: ...", try to extract
        m_url = re.search(r"(https?://[^\s'\"<>]+submit[^\s'\"<>]*)", final_text)
        if m_url:
            submit_url = m_url.group(1)
        # attempt to find a numeric answer
        m_ans = re.search(r"answer[:=]\s*([0-9\.\-Ee]+)", final_text, flags=re.I)
        if m_ans:
            answer_payload = {"email": original_payload.get("email"), "secret": original_payload.get("secret"), "url": original_payload.get("url"), "answer": float(m_ans.group(1))}
        # else leave as None

    return {"submit_url": submit_url, "answer_payload": answer_payload, "aipipe_final_text": final_text, "fn_result": fn_result, "selected_fn": selected_name}

# -------- Orchestrator for handling a full request (can follow multiple URLs until done or timeout) ----------
async def run_full_quiz_flow(initial_payload: Dict[str, Any], time_limit_seconds: int = TOTAL_TIME_LIMIT) -> Dict[str, Any]:
    """
    Orchestrate the entire flow within the time window:
      - load the initial URL
      - call analyze_page_and_solve to get submit_url & answer_payload
      - submit; if response indicates incorrect but provides next URL, repeat until timeout or no next URL
      - allow re-submission within 3 minutes (we'll resubmit if quiz returns incorrect and we still have time)
    """
    deadline = datetime.utcnow() + timedelta(seconds=time_limit_seconds)
    current_url = initial_payload.get("url")
    last_server_response = None
    history = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            while datetime.utcnow() < deadline and current_url:
                # load page
                await page.goto(current_url, wait_until="networkidle", timeout=30_000)
                # analyze and prepare submission
                solve_result = await analyze_page_and_solve(page, initial_payload, deadline)
                submit_url = solve_result.get("submit_url")
                answer_payload = solve_result.get("answer_payload")
                aid_text = solve_result.get("aipipe_final_text", "")

                if not submit_url or not answer_payload:
                    # we couldn't form a submission; abort and return debug info
                    return {"status": "failed", "reason": "no submit_url or answer_payload", "solve_debug": solve_result}

                # submit answer (allow re-submissions until deadline)
                async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                    resp = await client.post(submit_url, json=answer_payload)
                    # if non-2xx, return debug
                    try:
                        resp.raise_for_status()
                    except Exception:
                        return {"status": "submit_failed", "http_status": resp.status_code, "text": resp.text[:2000], "solve_debug": solve_result}

                    host_json = None
                    try:
                        host_json = resp.json()
                    except Exception:
                        host_json = {"text": resp.text}

                history.append({"url": current_url, "submit_url": submit_url, "answer_payload": answer_payload, "host_response": host_json, "time": now_iso()})
                last_server_response = host_json

                # If host says correct:true and provides next url -> follow it
                if isinstance(host_json, dict) and host_json.get("correct") is True:
                    next_url = host_json.get("url")
                    if next_url:
                        current_url = next_url
                        # continue loop to solve next
                        continue
                    else:
                        # no next url -> done
                        return {"status": "done", "history": history}
                else:
                    # incorrect or ambiguous -> check for next url or re-try within time
                    next_url = host_json.get("url") if isinstance(host_json, dict) else None
                    if next_url:
                        # host provided a next url even on incorrect -> follow it
                        current_url = next_url
                        continue
                    else:
                        # host didn't provide next url; if still time left, attempt to ask model for a revised answer:
                        # We'll allow one more attempt: call general_llm_solver with context that includes host_response
                        time_left = (deadline - datetime.utcnow()).total_seconds()
                        if time_left > 20:
                            # craft a retry: ask general LLM for corrected answer
                            retry_context = {
                                "last_host_response": host_json,
                                "previous_answer": answer_payload,
                                "page_url": current_url
                            }
                            # call general_llm_solver to get improved payload
                            retry_params = {"context": json.dumps(retry_context)[:4000], "instructions": "Produce a corrected answer_payload and submit_url if possible."}
                            retry_result = await general_llm_solver(retry_params)
                            # try to parse an answer_payload and submit_url from the LLM response (it may be AIPipe response)
                            # If it returned 'aipipe_response', attempt to parse content as JSON
                            parsed_submit_url = None
                            parsed_payload = None
                            if isinstance(retry_result, dict) and "aipipe_response" in retry_result:
                                final_text = ""
                                try:
                                    final_text = retry_result["aipipe_response"]["choices"][0]["message"].get("content","")
                                except Exception:
                                    final_text = str(retry_result["aipipe_response"])
                                candidate = re.search(r"\{.*\}", final_text, flags=re.S)
                                if candidate:
                                    try:
                                        j = json.loads(candidate.group(0))
                                        parsed_submit_url = j.get("submit_url")
                                        parsed_payload = j.get("answer_payload")
                                    except Exception:
                                        pass
                            # if parsed, set for next submission
                            if parsed_submit_url and parsed_payload:
                                submit_url = parsed_submit_url
                                answer_payload = parsed_payload
                                # ensure email/secret/url present
                                answer_payload.setdefault("email", initial_payload.get("email"))
                                answer_payload.setdefault("secret", initial_payload.get("secret"))
                                answer_payload.setdefault("url", initial_payload.get("url"))
                                # submit and continue loop (but do not change current_url unless host returns new one)
                                async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                                    resp2 = await client.post(submit_url, json=answer_payload)
                                    try:
                                        resp2.raise_for_status()
                                    except Exception:
                                        return {"status": "submit_failed_on_retry", "http_status": resp2.status_code, "text": resp2.text[:2000]}
                                    host_json2 = resp2.json() if resp2.headers.get("content-type","").startswith("application/json") else {"text": resp2.text}
                                history.append({"retry_submit_url": submit_url, "answer_payload": answer_payload, "host_response": host_json2, "time": now_iso()})
                                # check if now correct
                                if isinstance(host_json2, dict) and host_json2.get("correct") is True:
                                    next_url2 = host_json2.get("url")
                                    if next_url2:
                                        current_url = next_url2
                                        continue
                                    else:
                                        return {"status": "done", "history": history}
                                else:
                                    # still not correct and no next url -> stop
                                    return {"status": "failed_after_retry", "history": history}
                        # not enough time or no retry -> stop
                        return {"status": "failed_no_next", "history": history}

            # end while
            return {"status": "timeout_or_deadline", "history": history}
        finally:
            await context.close()
            await browser.close()

# -------- HTTP endpoint ----------
@app.post("/quiz", status_code=200)
async def quiz_endpoint(request: Request):
    # quick size check
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_PAYLOAD_BYTES:
                raise HTTPException(status_code=400, detail="Payload too large.")
        except Exception:
            pass

    # parse JSON
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # validate structure
    try:
        req = QuizRequest(**payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    # verify secret
    if req.secret != EXPECTED_SECRET:
        # 403 as required
        raise HTTPException(status_code=403, detail="Invalid secret.")

    # ack accepted
    response = {"status": "accepted", "received_at": now_iso(), "solver": "enabled"}

    # Run the full flow synchronously (must complete within TOTAL_TIME_LIMIT)
    try:
        result = await asyncio.wait_for(run_full_quiz_flow(payload, time_limit_seconds=TOTAL_TIME_LIMIT), timeout=TOTAL_TIME_LIMIT + 5)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="Solver timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {e}")

    response["result"] = result
    return response

# -------- Run note ----------
# To run:
# export SECRET="elephant"
# export AIPIPE_API_KEY="..."   # your key
# export AIPIPE_API_URL="https://api.aipipe.io/v1/chat/completions"
# uvicorn main:app --host 0.0.0.0 --port 8000
#
# Dependencies:
# pip install fastapi uvicorn[standard] playwright httpx pydantic
# python -m playwright install chromium
