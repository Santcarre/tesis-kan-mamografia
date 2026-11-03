import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from desktop_app.backend.config import APP_TITLE, VERSION, BASE_DIR
from desktop_app.backend.routers.health import router as health_router
from desktop_app.backend.routers.cohort import router as cohort_router
from desktop_app.backend.routers.cases import router as cases_router
from desktop_app.backend.routers.formula import router as formula_router
from desktop_app.backend.routers.inference import router as inference_router
from desktop_app.backend.routers.reports import router as reports_router

app = FastAPI(
    title=APP_TITLE,
    version=VERSION,
    description="Backend analítico para la estación radiológica neuro-simbólica Prometeo PACS v2.0",
)

# Configuración de CORS para permitir orígenes locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de routers
app.include_router(health_router)
app.include_router(cohort_router)
app.include_router(cases_router)
app.include_router(formula_router)
app.include_router(inference_router)
app.include_router(reports_router)



class NoCacheStaticFiles(StaticFiles):
    """Sirve la interfaz pidiendo al navegador que revalide siempre (ETag): así una
    versión nueva de la interfaz nunca queda oculta por la caché de Electron."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Serve static files
ui_dir = os.path.join(BASE_DIR.parent, "ui")
os.makedirs(ui_dir, exist_ok=True)
app.mount("/ui", NoCacheStaticFiles(directory=ui_dir, html=True), name="ui")

@app.get("/")
def redirect_to_ui():
    return RedirectResponse(url="/ui/index.html")

