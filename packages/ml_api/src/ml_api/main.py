"""Main file for running FastAPI application."""

import uvicorn

from ml_api.app import app


def main() -> None:
    """Main function to run FastAPI app with uvicorn."""
    uvicorn.run(app, host="127.0.0.1", port=5002)


if __name__ == "__main__":
    main()
