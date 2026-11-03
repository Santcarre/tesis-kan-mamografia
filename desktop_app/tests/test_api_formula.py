"""
Pruebas de la evaluación y el barrido de la fórmula canónica g(u0).

Valores de referencia de la tesis (Cap. 4, caso P_00141): g(+0,55) = -0,3395 y
p_sym = 41,6 %; umbral de Youden tau* = +0,1296 en logit (53,2 % en probabilidad).
"""
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_formula_eval_reproduce_valores_de_la_tesis():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/formula/eval", json={"u0": 0.549538})
        assert res.status_code == 200
        d = res.json()
        assert d["logit"] == pytest.approx(-0.3395, abs=2e-3)
        assert d["prob"] == pytest.approx(0.416, abs=2e-3)
        assert d["decision"] == "BENIGN"          # g < tau*
        assert d["youden_threshold"] == pytest.approx(0.1296, abs=1e-4)
        assert d["youden_threshold_prob"] == pytest.approx(0.532, abs=2e-3)
        assert d["birads_label"].startswith("BI-RADS 3")
        assert d["logit"] == pytest.approx(d["f_mal"] - d["f_ben"], abs=1e-4)
        assert "formula_expr" in d


@pytest.mark.asyncio
async def test_formula_decision_cambia_con_el_umbral():
    """g es decreciente en u0: u0 muy negativo => g alto => MALIGNANT; u0 muy positivo => BENIGN."""
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        low = (await client.post("/api/formula/eval", json={"u0": -2.0})).json()
        high = (await client.post("/api/formula/eval", json={"u0": 2.0})).json()
        assert low["decision"] == "MALIGNANT" and high["decision"] == "BENIGN"
        assert low["logit"] > high["logit"]


@pytest.mark.asyncio
async def test_formula_sweep_u0():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/formula/sweep?var_name=u0&val_min=-3.0&val_max=3.0&points=21")
        assert res.status_code == 200
        d = res.json()
        assert d["var_name"] == "u0"
        assert len(d["x_values"]) == len(d["prob_values"]) == len(d["logit_values"]) == 21
        assert len(d["fben_values"]) == len(d["fmal_values"]) == 21
        assert all(0.0 <= p <= 1.0 for p in d["prob_values"])
        # monotonía decreciente verificada en la tesis (g' < 0 en el soporte)
        g = d["logit_values"]
        assert all(g[i] > g[i + 1] for i in range(len(g) - 1))


@pytest.mark.asyncio
async def test_formula_sweep_rechaza_variables_que_no_son_u0():
    """No hay puente x -> u0: barrer x0, x1 o x2 debe devolver 400."""
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for var in ["x0", "x1", "x2", "invalid_var"]:
            res = await client.get(f"/api/formula/sweep?var_name={var}&val_min=-1.0&val_max=1.0&points=10")
            assert res.status_code == 400
