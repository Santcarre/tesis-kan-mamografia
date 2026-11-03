"""
Pruebas de integración para endpoints de cohorte y detalle de casos CBIS-DDSM.
"""
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_get_cohort_and_case_detail():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Test cohorte
        res_cohort = await client.get("/api/cohort?pathology=all&type=all")
        assert res_cohort.status_code == 200
        cohort = res_cohort.json()
        assert len(cohort) > 0
        assert "patient_id" in cohort[0]

        # Test caso individual (índice 0)
        res_case = await client.get("/api/cases/0")
        assert res_case.status_code == 200
        case_data = res_case.json()
        assert "prediction" in case_data
        assert "prob_malignant" in case_data["prediction"]
        assert "canonical_variables" in case_data
        assert "x0" in case_data["canonical_variables"]
        assert "x1" in case_data["canonical_variables"]
        assert "x2" in case_data["canonical_variables"]


@pytest.mark.asyncio
async def test_cohort_filtering():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Filtrar solo masas benignas
        res_benign_mass = await client.get("/api/cohort?pathology=benign&type=mass")
        assert res_benign_mass.status_code == 200
        cases = res_benign_mass.json()
        assert len(cases) > 0
        for c in cases:
            assert "BENIGN" in c["pathology"].upper()
            assert c["type"] == "mass"

        # Filtrar solo calcificaciones malignas
        res_mal_calc = await client.get("/api/cohort?pathology=malignant&type=calc")
        assert res_mal_calc.status_code == 200
        cases_mal = res_mal_calc.json()
        assert len(cases_mal) > 0
        for c in cases_mal:
            assert c["pathology"].upper() == "MALIGNANT"
            assert c["type"] == "calc"


@pytest.mark.asyncio
async def test_case_detail_not_found():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_case = await client.get("/api/cases/999999")
        assert res_case.status_code == 404
        data = res_case.json()
        assert "detail" in data
