"""
Launcher script for Vyuha ML FastAPI server.
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    print("Starting Vyuha ML FastAPI Service on http://127.0.0.1:8000 ...")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
