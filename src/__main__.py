"""
Entry point: python -m src
Starts the FastAPI server.
"""

import uvicorn


def main():
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
