import asyncio
import hashlib
import hmac
import os
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .inference import Classifier, DISPLAY_NAMES

WEB_ROOT = Path(__file__).resolve().parent.parent


class Query(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=1200)

    @field_validator('query')
    @classmethod
    def clean_query(cls, value):
        value = value.strip()
        if not value or any(ord(c) < 32 and c not in '\n\t\r' for c in value):
            raise ValueError('Escribe una consulta válida.')
        return value


class Explanation(Query):
    label: str = Field(min_length=1, max_length=100)

    @field_validator('label')
    @classmethod
    def known_label(cls, value):
        if value not in DISPLAY_NAMES:
            raise ValueError('Categoría desconocida.')
        return value


class RateLimit:
    """One-container guardrail, not a substitute for the provider's spend limit."""
    def __init__(self):
        self.calls = {'classify': deque(), 'explain': deque()}

    def take(self, action):
        now = time.monotonic()
        calls = self.calls[action]
        while calls and calls[0] <= now - 3600:
            calls.popleft()
        minute, hour = (12, 80) if action == 'classify' else (2, 12)
        if sum(x > now - 60 for x in calls) >= minute or len(calls) >= hour:
            raise HTTPException(429, 'Se alcanzó el límite de consultas. Inténtalo más tarde.', headers={'Retry-After': '60'})
        calls.append(now)


def create_app(classifier=None, explainer=None, *, deployment=False, access_hash=None, allowed_origins=None):
    classifier = classifier or Classifier(os.environ.get('BANKING77_MODEL_DIR', str(WEB_ROOT / 'model' / 'best')))
    expected_hash = access_hash if access_hash is not None else os.environ.get('B77_DEMO_ACCESS_SHA256', '')
    if deployment and (len(expected_hash) != 64 or any(c not in '0123456789abcdef' for c in expected_hash)):
        raise RuntimeError('Define B77_DEMO_ACCESS_SHA256 en un Secret del backend antes de desplegar.')
    app = FastAPI(title='Banking77 demonstration', docs_url=None, redoc_url=None, openapi_url=None)
    limiter = RateLimit()
    gate = asyncio.Semaphore(1)
    origins = allowed_origins or [x for x in os.environ.get('B77_ALLOWED_ORIGINS', '').split(',') if x]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'Authorization'], allow_credentials=False)

    def authorized(request):
        if not expected_hash:
            return
        auth = request.headers.get('authorization', '')
        code = auth[7:] if auth.startswith('Bearer ') else ''
        actual = hashlib.sha256(code.encode()).hexdigest()
        if not code or not hmac.compare_digest(actual, expected_hash):
            raise HTTPException(401, 'Código de acceso incorrecto. Solicita el código a Enrique.')

    @app.middleware('http')
    async def guard(request: Request, call_next):
        if request.url.path.startswith('/api/') and request.method == 'POST':
            raw = await request.body()
            if len(raw) > 8192:
                return JSONResponse({'detail': 'La consulta es demasiado larga.'}, status_code=413)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store' if request.url.path.startswith('/api/') else 'no-cache'
        response.headers['Content-Security-Policy'] = "default-src 'self'; connect-src 'self' https://*.modal.run; img-src 'self' data:; style-src 'self'; font-src 'self'; script-src 'self'; base-uri 'self'; form-action 'self'"
        return response

    @app.get('/api/health')
    async def health():
        is_ready = classifier.ready
        return {'classifier_ready': is_ready, 'falcon_available': explainer is not None,
                'requires_access_code': bool(expected_hash),
                'message': '' if is_ready else 'Falta conectar el modelo entrenado. La página está lista; no se mostrarán predicciones simuladas.'}

    @app.post('/api/classify')
    async def classify(body: Query, request: Request):
        authorized(request)
        if not classifier.ready:
            raise HTTPException(503, 'Falta conectar el modelo entrenado. No se puede clasificar todavía.')
        limiter.take('classify')
        async with gate:
            try:
                return await asyncio.to_thread(classifier.predict, body.query)
            except Exception:
                raise HTTPException(503, 'No se pudo cargar o ejecutar el modelo. Inténtalo más tarde.') from None

    @app.post('/api/explain')
    async def explain(body: Explanation, request: Request):
        authorized(request)
        if not classifier.ready or explainer is None:
            raise HTTPException(503, 'Falcon aún no está conectado a esta demo.')
        limiter.take('explain')
        async with gate:
            try:
                prediction = await asyncio.to_thread(classifier.predict, body.query)
                if prediction['label'] != body.label:
                    raise HTTPException(409, 'La categoría ya no coincide con la consulta. Clasifica de nuevo.')
                if asyncio.iscoroutinefunction(explainer):
                    return await explainer(body.query, body.label)
                return await asyncio.to_thread(explainer, body.query, body.label)
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(503, 'Falcon no pudo completar la generación. Inténtalo más tarde.') from None

    @app.get('/')
    async def index():
        return FileResponse(WEB_ROOT / 'index.html')

    @app.get('/{asset:path}')
    async def asset(asset: str):
        # Never expose the checkpoint, backend, tests, evidence or credentials.
        if asset not in {'styles.css', 'app.js', 'config.js'} and not asset.startswith('assets/'):
            raise HTTPException(404)
        file = (WEB_ROOT / asset).resolve()
        if not file.is_relative_to(WEB_ROOT) or not file.is_file():
            raise HTTPException(404)
        return FileResponse(file)

    return app


app = create_app()
