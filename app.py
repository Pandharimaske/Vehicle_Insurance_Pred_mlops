from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run
from pydantic import BaseModel, Field
from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator

# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vehicle Insurance Claim Prediction API",
    description=(
        "Binary classification API predicting vehicle insurance claim likelihood. "
        "Powered by XGBoost/LightGBM trained with Optuna HPO (50 trials, TPE sampler). "
        "Experiments tracked on DagsHub via MLflow."
    ),
    version="1.0.0",
)

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Allow all origins for Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas — Bullet 4: input validation + structured JSON responses
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Validated input schema for the /predict JSON endpoint.
    All fields carry type constraints; FastAPI returns a 422 with detail
    if any constraint is violated — no silent bad predictions.
    """
    Gender: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male")
    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Driving_License: int = Field(..., ge=0, le=1, description="1 = has driving licence")
    Region_Code: float = Field(..., ge=0.0, description="Encoded region code")
    Previously_Insured: int = Field(..., ge=0, le=1, description="1 = was previously insured")
    Annual_Premium: float = Field(..., gt=0.0, description="Annual insurance premium (₹)")
    Policy_Sales_Channel: float = Field(..., gt=0.0, description="Encoded sales channel")
    Vintage: int = Field(..., ge=0, description="Days customer has been associated with company")
    Vehicle_Age_lt_1_Year: int = Field(..., ge=0, le=1, description="1 = vehicle age < 1 year")
    Vehicle_Age_gt_2_Years: int = Field(..., ge=0, le=1, description="1 = vehicle age > 2 years")
    Vehicle_Damage_Yes: int = Field(..., ge=0, le=1, description="1 = vehicle was previously damaged")

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": 1,
                "Age": 35,
                "Driving_License": 1,
                "Region_Code": 28.0,
                "Previously_Insured": 0,
                "Annual_Premium": 40454.0,
                "Policy_Sales_Channel": 26.0,
                "Vintage": 127,
                "Vehicle_Age_lt_1_Year": 0,
                "Vehicle_Age_gt_2_Years": 1,
                "Vehicle_Damage_Yes": 1,
            }
        }


class PredictResponse(BaseModel):
    """Structured JSON response from the /predict endpoint."""
    prediction: int = Field(..., description="0 = No claim, 1 = Likely to claim")
    label: str = Field(..., description="Human-readable prediction label")
    confidence: float = Field(..., description="Model probability for the predicted class (0–1)")


# ---------------------------------------------------------------------------
# HTML form helper (used by the legacy / route)
# ---------------------------------------------------------------------------

class DataForm:
    """Reads raw form fields from HTML form submission."""
    def __init__(self, request: Request):
        self.request: Request = request
        self.Gender: Optional[int] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age_lt_1_Year: Optional[int] = None
        self.Vehicle_Age_gt_2_Years: Optional[int] = None
        self.Vehicle_Damage_Yes: Optional[int] = None

    async def get_vehicle_data(self):
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Age = form.get("Age")
        self.Driving_License = form.get("Driving_License")
        self.Region_Code = form.get("Region_Code")
        self.Previously_Insured = form.get("Previously_Insured")
        self.Annual_Premium = form.get("Annual_Premium")
        self.Policy_Sales_Channel = form.get("Policy_Sales_Channel")
        self.Vintage = form.get("Vintage")
        self.Vehicle_Age_lt_1_Year = form.get("Vehicle_Age_lt_1_Year")
        self.Vehicle_Age_gt_2_Years = form.get("Vehicle_Age_gt_2_Years")
        self.Vehicle_Damage_Yes = form.get("Vehicle_Damage_Yes")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["UI"])
async def index(request: Request):
    """Renders the main HTML form page for vehicle data input."""
    return templates.TemplateResponse(
        "vehicledata.html", {"request": request, "context": "Rendering"}
    )


@app.get("/train", tags=["training"])
async def train_route():
    """
    Triggers the full training pipeline:
      data ingestion → validation → transformation
      → Optuna HPO (XGBoost/LightGBM, 50 trials) → champion/challenger evaluation
      → MLflow/DagsHub logging → S3 model push.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training pipeline completed successfully.")
    except Exception as e:
        return Response(f"Error during training: {e}")


@app.post("/", tags=["UI"], include_in_schema=False)
async def predict_route_client(request: Request):
    """Legacy HTML-form prediction endpoint (returns rendered HTML page)."""
    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        vehicle_data = VehicleData(
            Gender=form.Gender,
            Age=form.Age,
            Driving_License=form.Driving_License,
            Region_Code=form.Region_Code,
            Previously_Insured=form.Previously_Insured,
            Annual_Premium=form.Annual_Premium,
            Policy_Sales_Channel=form.Policy_Sales_Channel,
            Vintage=form.Vintage,
            Vehicle_Age_lt_1_Year=form.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=form.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=form.Vehicle_Damage_Yes,
        )
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()
        model_predictor = VehicleDataClassifier()
        value = model_predictor.predict(dataframe=vehicle_df)[0]
        status = "Response-Yes" if value == 1 else "Response-No"

        return templates.TemplateResponse(
            "vehicledata.html",
            {"request": request, "context": status},
        )
    except Exception as e:
        return {"status": False, "error": f"{e}"}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict_api(request: PredictRequest):
    """
    Structured JSON inference endpoint (Bullet 4).

    Accepts a validated JSON body (PredictRequest), runs the model,
    and returns a structured JSON response with:
      - `prediction`  — 0 (no claim) or 1 (likely to claim)
      - `label`       — "Response-Yes" or "Response-No"
      - `confidence`  — probability of the positive class (0–1)

    Input validation is handled automatically by Pydantic; invalid
    requests receive a 422 Unprocessable Entity with field-level errors.
    """
    try:
        vehicle_data = VehicleData(
            Gender=request.Gender,
            Age=request.Age,
            Driving_License=request.Driving_License,
            Region_Code=request.Region_Code,
            Previously_Insured=request.Previously_Insured,
            Annual_Premium=request.Annual_Premium,
            Policy_Sales_Channel=request.Policy_Sales_Channel,
            Vintage=request.Vintage,
            Vehicle_Age_lt_1_Year=request.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=request.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=request.Vehicle_Damage_Yes,
        )
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # Load model from S3 and get probability + binary prediction
        predictor_config = VehiclePredictorConfig()
        estimator = Proj1Estimator(
            bucket_name=predictor_config.model_bucket_name,
            model_path=predictor_config.model_file_path,
        )
        proba = estimator.predict_proba(vehicle_df)[0]   # shape (2,)
        confidence = float(proba[1])                      # P(claim)
        prediction = int(confidence >= 0.5)
        label = "Response-Yes" if prediction == 1 else "Response-No"

        return PredictResponse(
            prediction=prediction,
            label=label,
            confidence=round(confidence, 4),
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)