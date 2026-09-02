from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import predict_irrigation

app = FastAPI(title="AI Irrigation System")
templates = Jinja2Templates(directory="templates")

# Page 1: Introduction Landing Page
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse(request, "landing.html")

# Page 2: Manual Form Page
@app.get("/analyze", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse(request, "form.html")

# Page 3: Prediction Result Page
@app.post("/predict", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    first_merging: str = Form(...),
    irrigation_needed: str = Form(...),
    region: str = Form(...),
    water_source: str = Form(...),
    irrigation_method: str = Form(...),
    season: str = Form(...),
    growth_stage: str = Form(...),
    crop_type: str = Form(...),
    soil_type: str = Form(...),
    sunlight_hours: str = Form(...),
    temperature: str = Form(...)
):
    captured_data = {
        "First Merging Used": first_merging,
        "Irrigation Needed": irrigation_needed,
        "Region": region,
        "Water Source": water_source,
        "Irrigation Method": irrigation_method,
        "Season": season,
        "Crop Growth Stage": growth_stage,
        "Crop Type": crop_type,
        "Soil Type": soil_type,
        "Sunlight Hours": f"{sunlight_hours} hrs",
        "Temperature": f"{temperature}°C"
    }
    
    # Real ML prediction using model.py
    raw_input_data = {
        "first_merging": first_merging,
        "region": region,
        "water_source": water_source,
        "irrigation_method": irrigation_method,
        "season": season,
        "growth_stage": growth_stage,
        "crop_type": crop_type,
        "soil_type": soil_type,
        "sunlight_hours": sunlight_hours,
        "temperature": temperature
    }
    
    prediction = predict_irrigation(raw_input_data)

    return templates.TemplateResponse(
        request, 
        "result.html", 
        {"data": captured_data, "prediction": prediction}
    )