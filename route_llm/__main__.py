"""python -m route_llm entry point."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("route_llm.main:app", host="0.0.0.0", port=8000, reload=True)
