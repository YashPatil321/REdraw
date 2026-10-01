"""FastAPI app (`api.main:app`). Endpoints are documented in docs/api.md.

Run: `.venv/bin/uvicorn api.main:app --reload`
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api import routes_plans, routes_residents, routes_world
from api.db import create_schema, make_engine
from api.jobs import JobRunner
from api.settings import Settings, get_settings
from api.state import AppState, ServiceUnavailable, SimFactory, default_sim_factory
from api.store import Store
from residents.llm import ChatLLM, LLMClient, NullLLM

log = logging.getLogger("api")

PLAYER_COOKIE = "redraw_player"
COOKIE_MAX_AGE_S = 60 * 60 * 24 * 365 * 2


def _valid_uuid(v: str | None) -> bool:
    if not v:
        return False
    try:
        uuid.UUID(v)
    except ValueError:
        return False
    return True


def make_llm(settings: Settings) -> ChatLLM:
    if not settings.llm_base_url:
        return NullLLM()
    return LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model_fast,
                     settings.llm_model_smart, timeout_s=settings.llm_timeout_s)


def create_app(settings: Settings | None = None, sim_factory: SimFactory | None = None,
               llm: ChatLLM | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.sqlalchemy_url)
    store = Store(engine)
    state = AppState(settings, store, sim_factory or default_sim_factory, llm or make_llm(settings))
    runner = JobRunner(state, settings.redraw_job_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        create_schema(engine)
        n = store.fail_stale_jobs()
        if n:
            log.warning("marked %d unfinished jobs from a previous run as failed", n)
        if settings.redraw_warm_baseline:
            state.start_warmup()
        yield
        runner.shutdown()

    app = FastAPI(title="Redraw API", version="1", lifespan=lifespan)
    app.state.redraw = state
    app.state.jobs = runner

    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def player_cookie(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        pid = request.cookies.get(PLAYER_COOKIE)
        new = not _valid_uuid(pid)
        if new:
            pid = str(uuid.uuid4())
        request.state.player_id = pid
        response = await call_next(request)
        if new:
            response.set_cookie(PLAYER_COOKIE, pid or "", max_age=COOKIE_MAX_AGE_S, httponly=True,
                                samesite="lax", secure=settings.cookie_secure, path="/")
        return response

    @app.exception_handler(ServiceUnavailable)
    async def _unavailable(_: Request, exc: ServiceUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, object]:
        frac, msg = state.baseline_progress
        return {"ok": True, "baseline_ready": state.baseline_summary is not None,
                "baseline_progress": frac, "baseline_message": msg, "baseline_error": state.baseline_error,
                "llm_available": bool(state.llm.available)}

    app.include_router(routes_world.router)
    app.include_router(routes_plans.router)
    app.include_router(routes_residents.router)

    settings.assets_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(settings.assets_dir), check_dir=False), name="assets")
    return app


def __getattr__(name: str) -> FastAPI:
    # Build the module-level `app` lazily so importing api.main (e.g. in tests) has no side effects.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
