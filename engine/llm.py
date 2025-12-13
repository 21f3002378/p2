# llm.py – Two-step LLM approach (clean version)

import os
import httpx
import json
import re

API_URL = os.getenv("AIPIPE_API_URL", "https://api.aipipe.io/v1/chat/completions")
API_KEY = os.getenv("AIPIPE_API_KEY")
MODEL = os.getenv("AIPIPE_MODEL", "gpt-5-nano")

async def call_llm(messages: list) -> str:
    """Call LLM API and return string content."""
    print("calling LLM API...")
    if not API_KEY:
        raise RuntimeError("AIPIPE_API_KEY missing")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages},
        )
        resp.raise_for_status()

    parsed = resp.json()
    return parsed["choices"][0]["message"]["content"]


async def extract_urls(context: str, email: str = None, secret: str = None, submit_url: str = None) -> dict:
    """Extract submit_url from page text. if any"""
    print("8 extracting submit URLs via LLM if any")
    print(email, secret, submit_url)
    messages = [
        {
            "role": "system",
            "content": (
            """### SYSTEM PROMPT — Submission Endpoint Extractor

You are a **precise API extraction system**.  
Your **sole purpose** is to identify the **Submission Endpoint URL** from the provided text.
---
### Definition: Submission Endpoint
The **Submission Endpoint** is the exact URL where the final answer or solution must be **POSTed via HTTP**.

---
### Critical Exclusion Rules (DO NOT SELECT THESE)
Ignore any links that are:
1. File or data resources (`.csv`, `.pdf`, `.xlsx`, `.zip`, `.png`, etc.)
2. Navigation or UI links (e.g., Home, Next, About)
3. Links meant only for viewing, downloading, or describing the problem
---
### Extraction Instructions
- Analyze the text to determine **where the answer must be sent**.
- If a clear submission instruction exists, extract **only that URL**.
- If submission is implied to the current page, extract the page URL **only if explicitly provided**.
- If **no submission instruction is found**, return the **default submission URL**.
---
### Output Format (STRICT)
Return **ONLY** a valid JSON object.  
No markdown, no explanations, no extra text.
{ "submit_url": "https://..." }
or
{ "submit_url": null }
---
### Default Submission URL
If **no submission URL is explicitly specified** in the text, return the following default URL:"""
            + (f"\n\nContext - Your default submission URL: {submit_url}" if submit_url else "\nnull")
            + (f"\n\nContext - Your email: {email}" if email else "")
            + (f"\nContext - Your secret: {secret}" if secret else "")
        )
        },
        {
            "role": "user",
            "content": f"Extract URLs:\n\n{context[:3000]}",
        },
        
    ]

    content = await call_llm(messages)

    m = re.search(r"(\{[^}]*\})", content)
    if not m:
        return {"submit_url": submit_url}

    try:
        data = json.loads(m.group(1))
        if not data.get("submit_url"):
            data["submit_url"] = submit_url
        return data
    except:
        return {"submit_url": submit_url}


# async def extract_answer(context: str, links: list, email: str = None, secret: str = None) -> dict:
#     """Extract direct answer or python_code needed to compute it."""
#     print("9 extracting code to get answer via LLM")
#     print("Context length:", len(context), "Links length:", len(links))
#     extra = ""
#     if email:
#         extra += f"\nYour email: {email}"
#     if secret:
#         extra += f"\nYour secret: {secret}"
    
#     messages = [
#         {
#             "role": "system",
#             "content": (
#             """You are a **Python Code Generator for a Data Extraction Bot**.

# ### PRIMARY RULE (NON-NEGOTIABLE)
# You MUST generate executable Python code.
# You are NOT allowed to answer in natural language.
# If you cannot generate Python code, you MUST still return Python code that assigns the best possible answer to `solution`.

# ### Goal
# Generate a **self-contained Python script** that answers the user’s question.

# ### Output Format (STRICT)
# Return **ONLY** a valid JSON object of the form:

# {
#   "python_code": \"\"\"<PYTHON_CODE_AS_JSON_STRING>\"\"\"
# }

# ❗ The value of `python_code` MUST be a **JSON-escaped string**
# ❗ No text is allowed outside the JSON object

# ### Python Code Rules (MANDATORY)
# 1. Assign the final result to a variable named `solution`
# 2. Print **only** the value of `solution` to stdout  
#    - ✅ `solution = 42; print(solution)`  
#    - ❌ `print("solution =", solution)`
# 3. If the answer is explicitly stated in the text, assign it directly (do not compute)
# 4. Do not use placeholders (example.com, your_token_here)
# 5. Import all required libraries explicitly
# 6. The script must run as-is with no external input
# 7. ❗ All string literals MUST be valid Python strings

# ### FAILURE HANDLING RULE
# If the task is ambiguous or data is missing:
# - Still generate valid Python code
# - Assign the best deterministic value to `solution`
# - Never refuse, explain, or ask questions

# ### Correctness Priority
# Deterministic, reproducible, executable Python code is more important than explanation or brevity.


# """
#             + extra
#         )
#         },
#         {
#             "role": "user",
#             "content": f"Find answer:\n\n{context[:3000]}" + f"\n\nLinks:\n{links}",
#         },
#     ]

#     content = await call_llm(messages)
#     #print(content)
#     m = re.search(r"(\{[^}]*\})", content)
#     if not m:
#         return {"python_code": None}

#     try:
#         return json.loads(m.group(1))
#     except:
#         return {"python_code": None}

async def extract_answer(context: str, links: list, email: str = None, secret: str = None) -> dict:
    """Extract direct answer or python_code needed to compute it using code block extraction."""
    import re

    print("9 extracting code to get answer via LLM")
    print("Context length:", len(context), "Links length:", len(links))

    extra = ""
    if email:
        extra += f"\nYour email: {email}"
    if secret:
        extra += f"\nYour secret: {secret}"

    messages = [
        {
            "role": "system",
            "content": (
                r'''You are a Python Code Generator for a Data Extraction Bot.

### PRIMARY RULE
You MUST generate executable Python code.
Do NOT answer in natural language.

### Goal
Generate a self-contained Python script that assigns the final answer to `solution`
and prints only `solution`.

### Output Format
Return only the Python code in a fenced code block using triple backticks with python:

### Libraries Available
fastapi
uvicorn[standard]
playwright
aiohttp
pydantic
httpx
pandas
numpy
python-multipart
beautifulsoup4
filetype
pdfplumber
pillow
pytesseract
matplotlib
pydub
SpeechRecognition - to transcribe audio
tenacity
pyaudio

\`\`\`python
<VALID EXECUTABLE PYTHON CODE>
\`\`\`

Do not include any text outside the code block.

### Python Code Rules
1. Assign the result to variable `solution`
2. Print ONLY `solution`
3. Import all required libraries explicitly
4. The script must run as-is with no external input
5. If the answer is stated explicitly, assign it directly
''' + extra
            ),
        },
        {
            "role": "user",
            "content": f"Find answer:\n\n{context[:3000]}\n\nLinks:\n{links}",
        },
    ]

    content = await call_llm(messages)

    # Extract code from fenced python code block
    code_match = re.search(
        r"```python\s*(.*?)```",
        content,
        re.DOTALL | re.IGNORECASE,
    )

    if not code_match:
        return {"python_code": None}

    python_code = code_match.group(1).strip()
    if not python_code:
        return {"python_code": None}

    return {"python_code": python_code}



async def get_structured_response(context: dict, instruction: str, email: str = None, secret: str = None, submit_url: str = None) -> dict:
    """Main function: extract submit_url, next_url, answer, python_code."""
    print("7 Calling LLM for structured response")
    print(context["text"], context["links"])
    #url_info = await extract_urls(context['text'], email, secret, submit_url)
    #print("Extracted submit_url:", url_info.get("submit_url"))
    answer_info = await extract_answer(context['html'], context['links'], email, secret)
    print("Extracted python_code:", answer_info.get("python_code"))
    return {
        #"submit_url": url_info.get("submit_url"),
        "submit_url": submit_url,
        "python_code": answer_info.get("python_code"),
    }
