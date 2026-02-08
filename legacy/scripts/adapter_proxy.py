import argparse, os, pathlib, httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()
client = httpx.AsyncClient()

def touch_adapter(adapter_id: str, tier2_path: str):
    if not adapter_id:
        return
    path = pathlib.Path(tier2_path) / "lora" / adapter_id
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "ab") as f:
            f.write(b"\0" * 1048576)  # 1 MiB write
        with open(path, "rb") as f:
            f.read(1048576)          # 1 MiB read
    except Exception:
        pass


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    adapter_id = body.pop("adapter_id", None)  # strip before forwarding
    touch_adapter(adapter_id, os.environ.get("DYN_KVBM_TIER2_PATH", "/nvme/kvbm/l70b"))
    try:
        resp = await client.post(
            os.environ["BACKEND_URL"],
            headers={"content-type": "application/json"},
            json=body,
            timeout=120,
        )
    except Exception as exc:
        return {"error": f"backend_request_failed: {exc}"}

    # If backend returns JSON, forward it; otherwise forward raw text/status.
    try:
        return resp.json()
    except Exception:
        return JSONResponse(
            status_code=resp.status_code,
            content={"status_code": resp.status_code, "text": resp.text, "error": "non_json_response"},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()
    uvicorn.run(app, host=args.listen, port=args.port)
