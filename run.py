#!/usr/bin/env python3
"""
Convenience launcher for the Vehicle Damage Assessment & Insurance Claim Assistance application.
Starts the FastAPI backend and serves the frontend surveyor portal.
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting FastAPI Server & Surveyor Web Portal at http://localhost:8000 ...")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
