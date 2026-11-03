from fastapi.testclient import TestClient

from desktop_app.backend.app import app

client = TestClient(app)


def test_ui_index():
    response = client.get("/ui/index.html")
    assert response.status_code == 200
    assert "Prometeo" in response.text


def test_root_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/index.html"


def test_ui_sirve_modulos_y_datos_del_laboratorio():
    # La interfaz carga sus módulos y datos con rutas relativas bajo /ui/.
    for ruta in ("js/app.js", "js/lab.js", "css/lab.css",
                 "data/tesis.json", "data/lab_seleccion.json", "data/lab_pacientes.json", "data/lab_global.json"):
        r = client.get(f"/ui/{ruta}")
        assert r.status_code == 200, ruta
    assert client.get("/ui/data/tesis.json").json()["formula"]["f_mal"]


def test_ui_pide_revalidar_la_cache():
    # Evita que Electron muestre una versión anterior de la interfaz.
    r = client.get("/ui/index.html")
    assert r.headers.get("cache-control") == "no-cache"
