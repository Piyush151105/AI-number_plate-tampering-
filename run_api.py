import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from config.settings import settings

if __name__ == "__main__":
    # reload=False keeps the server stable on Windows.
    # The terminal must stay open while you use the dashboard.
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
