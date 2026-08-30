"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from abtestlab import __version__
from abtestlab.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    """Build the ASGI app with API routes mounted.

    Returns:
        Configured :class:`~fastapi.FastAPI` instance.
    """
    app = FastAPI(
        title="abtestlab",
        description=(
            "Local experimentation API: power analysis, fixed-horizon and sequential "
            "(mSPRT) tests, CUPED variance reduction, and Bayesian "
            "probability-to-beat-control reports."
        ),
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
