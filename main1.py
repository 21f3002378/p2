from dotenv import load_dotenv
load_dotenv()

import asyncio
from fastapi import FastAPI, Request, HTTPException
from engine.orchestrator import run_quiz_flow

EXPECTED_SECRET = "lolipop"

app = FastAPI()


@app.post("/quiz")
async def quiz_endpoint(request: Request):
    print("1 Received /quiz request")
    # ---- JSON VALIDATION ----
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        payload = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # ---- SECRET CHECK ----
    if payload.get("secret") != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    print("2 Valid request")

    # ---- QUIZ SOLVER ----
    task = asyncio.create_task(run_quiz_flow(payload))
    done, pending = await asyncio.wait({task}, timeout=185)

    if task not in done:
        raise HTTPException(status_code=500, detail="Timed out")

    result = task.result()

    return {"status": "done", "result": result}

