## TODOs

- Retest Stack A custom (larger context) build path using the updated `setup_triton_server_llama.sh` to ensure 16k+ admits still build cleanly.
- Stage 1 (session_id accept): Extend Dynamo OpenAI frontend request structs to accept an optional `session_id` so the API stops 400-ing when sessioned_chat is used, even if the value is ignored in routing.
- Stage 2 (session_id KV reuse): Plumb `session_id` through preprocessor/router/connector to key KV blocks so H9 can exercise real re-hydration (build→idle→resume with tier2 reads/TTFT bump).
