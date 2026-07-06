"""
Half-Cell Battery Simulator - Backend
Standalone FastAPI service running PyBaMM DFN half-cell simulations.

Run locally:
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000

Deploy on Render.com:
    Build command: pip install -r requirements.txt
    Start command: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pybamm
import numpy as np

app = FastAPI(title="Half-Cell Battery Simulator API")

# Allow the frontend (served from anywhere - file:// or any static host) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HalfCellRequest(BaseModel):
    working_electrode: Literal["positive", "negative"] = "positive"
    parameter_set: str = "Xu2019"
    c_rate: float = Field(default=0.5, gt=0, le=10, description="Discharge/charge C-rate")
    discharge_cutoff_v: Optional[float] = None
    charge_cutoff_v: Optional[float] = None
    initial_soc: float = Field(default=1.0, ge=0, le=1)


# Sensible default voltage windows per electrode (vs Li/Li+ reference)
DEFAULT_VOLTAGE_WINDOWS = {
    "positive": {"discharge_cutoff_v": 3.5, "charge_cutoff_v": 4.2},
    "negative": {"discharge_cutoff_v": 0.01, "charge_cutoff_v": 1.5},
}


@app.get("/")
def root():
    return {"status": "ok", "service": "half-cell-simulator", "pybamm_version": pybamm.__version__}


@app.get("/parameter-sets")
def list_parameter_sets():
    """Return commonly used parameter sets compatible with half-cell DFN models."""
    return {
        "parameter_sets": ["Xu2019", "Chen2020", "Marquis2019", "Ai2020"]
    }


@app.post("/simulate/half-cell")
def simulate_half_cell(req: HalfCellRequest):
    try:
        defaults = DEFAULT_VOLTAGE_WINDOWS[req.working_electrode]
        discharge_v = req.discharge_cutoff_v if req.discharge_cutoff_v is not None else defaults["discharge_cutoff_v"]
        charge_v = req.charge_cutoff_v if req.charge_cutoff_v is not None else defaults["charge_cutoff_v"]

        model = pybamm.lithium_ion.DFN({"working electrode": req.working_electrode})
        params = pybamm.ParameterValues(req.parameter_set)

        experiment = pybamm.Experiment(
            [
                f"Discharge at {req.c_rate}C until {discharge_v} V",
                f"Charge at {req.c_rate}C until {charge_v} V",
            ]
        )

        sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
        sol = sim.solve(initial_soc=req.initial_soc)

        time_s = sol["Time [s]"].entries.tolist()
        try:
            voltage_v = sol["Voltage [V]"].entries.tolist()
        except Exception:
            voltage_v = sol["Terminal voltage [V]"].entries.tolist()

        # Discharge capacity may not exist for every model variant; guard for it
        try:
            capacity_ah = sol["Discharge capacity [A.h]"].entries.tolist()
        except Exception:
            capacity_ah = []

        return {
            "working_electrode": req.working_electrode,
            "parameter_set": req.parameter_set,
            "c_rate": req.c_rate,
            "discharge_cutoff_v": discharge_v,
            "charge_cutoff_v": charge_v,
            "time_s": time_s,
            "voltage_v": voltage_v,
            "capacity_ah": capacity_ah,
        }

    except pybamm.SolverError as e:
        raise HTTPException(status_code=422, detail=f"Solver failed to converge: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter set or missing variable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DegradationRequest(BaseModel):
    sei_exchange_current_density: float = Field(
        default=1.5e-07, gt=0, description="SEI reaction exchange current density [A.m-2]"
    )
    c_rate: float = Field(default=0.3, gt=0, le=5)
    charge_cutoff_v: float = Field(default=1.5)
    discharge_cutoff_v: float = Field(default=0.005)


@app.post("/simulate/degradation")
def simulate_degradation(req: DegradationRequest):
    """
    Graphite-silicon composite half-cell degradation model (O'Kane et al.).
    Mirrors the coupled SEI / lithium-plating experiment from the reference notebook.
    """
    try:
        model = pybamm.lithium_ion.DFN(
            {
                "working electrode": "positive",
                "SEI": "reaction limited",
                "SEI porosity change": "true",
                "particle mechanics": "swelling and cracking",
                "SEI on cracks": "true",
                "lithium plating": "partially reversible",
                "lithium plating porosity change": "true",
            }
        )
        params = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
        params.update({"SEI reaction exchange current density [A.m-2]": req.sei_exchange_current_density})
        var_pts = {"x_n": 1, "x_s": 5, "x_p": 7, "r_n": 1, "r_p": 30}

        experiment = pybamm.Experiment(
            [
                f"Charge at {req.c_rate}C until {req.charge_cutoff_v} V",
                f"Discharge at {req.c_rate}C until {req.discharge_cutoff_v} V",
            ]
        )
        sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment, var_pts=var_pts)
        solver = pybamm.IDAKLUSolver()
        sol = sim.solve(solver=solver)

        t = sol["Time [s]"].entries.tolist()
        try:
            V = sol["Voltage [V]"].entries.tolist()
        except Exception:
            V = sol["Terminal voltage [V]"].entries.tolist()

        return {
            "time_s": t,
            "voltage_v": V,
            "loss_negative_sei_ah": sol["Loss of capacity to negative SEI [A.h]"].entries.tolist(),
            "loss_positive_sei_ah": sol["Loss of capacity to positive SEI [A.h]"].entries.tolist(),
            "loss_sei_on_cracks_ah": sol["Loss of capacity to positive SEI on cracks [A.h]"].entries.tolist(),
            "loss_lithium_plating_ah": sol["Loss of capacity to positive lithium plating [A.h]"].entries.tolist(),
            "sei_exchange_current_density": req.sei_exchange_current_density,
        }

    except pybamm.SolverError as e:
        raise HTTPException(status_code=422, detail=f"Solver failed to converge: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing variable (check model options): {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
