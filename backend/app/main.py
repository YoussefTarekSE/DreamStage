import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .errors import safe_error_detail
from .logging_config import configure_logging, request_logging_middleware
from .routers import health, projects, voice_training, studio, beat, coach, mix, admin

configure_logging()

app = FastAPI(title="DreamStage API", version="0.1.0")

app.middleware("http")(request_logging_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://dreamstage.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(projects.router, prefix="/api")
app.include_router(voice_training.router, prefix="/api")
app.include_router(studio.router, prefix="/api")
app.include_router(beat.router, prefix="/api")
app.include_router(coach.router, prefix="/api")
app.include_router(mix.router, prefix="/api")
app.include_router(admin.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": safe_error_detail(
                reason="internal_error",
                message_en="Something went wrong. Please try again.",
                message_ar="Something went wrong. Please try again.",
                debug=traceback.format_exc(),
            )
        },
    )
