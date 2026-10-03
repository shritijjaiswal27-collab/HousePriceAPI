import io
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field

app=FastAPI()
model=joblib.load(("house_model.joblib"))
features=joblib.load("housea_features.joblib")

class HouseFeatures(BaseModel):
    MedInc: float=Field(gt=0,description="Media Income of" \
    "Neighborhood")
    HouseAge: float=Field(gt=0, description="Average age of the house in the block")
    AveRooms: float=Field(gt=0, description="Average number of rooms in the house")
    AveBedrms: float=Field(gt=0, description="Average number of bedrooms in the house")
    Population: float=Field(gt=0, description="Population of the block")
    AveOccup: float=Field(gt=0, description="Average number of occupants in the house")
    Latitude: float=Field(gt=0, description="Latitude of the block")
    Longitude: float=Field(gt=0, description="Longitude of the block")

@app.get("/")
def home():
    return{
        "message":"California House Price Prediction API",
        "status":"running",
        "endpoint":"send POST request to /predict"
    }

@app.get("/health")
def health():
    return{
        "status":"running",
        "model":"Random Forest Regressor",
        "features":features,
        "avg error":"$ 40,000",
        "endpoint":"send POST request to /predict"
    }

@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data=pd.DataFrame([{
            "MedInc":house.MedInc,
            "HouseAge":house.HouseAge,
            "AveRooms":house.AveRooms,
            "AveBedrms":house.AveBedrms,
            "Population":house.Population,
            "AveOccup":house.AveOccup,
            "Latitude":house.Latitude,
            "Longitude":house.Longitude
        }])

        predicted_price=model.predict(input_data)[0]
        price_usd=predicted_price*100000
        
        return{
            "predicted_price":f"${price_usd:,.0f}",
            "predicted_price_short":f"${predicted_price:,.2f} hundred thousands",
            "fidence_range":f"${price_usd-39000:,.0f} to ${price_usd+39000:,.0f}"
        
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error occurred while making prediction: {str(e)}")


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):


    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    contents=await file.read()

    df = pd.read_csv(io.BytesIO(contents))

    required_columns=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    missing_columns=[col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_columns)}")

    if len(df)==0:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    try:
        predictions=model.predict(df[required_columns])
        df["PredictedPrice"]=[f"${pred*100000:,.0f}" for pred in predictions]

        output=io.StringIO()
        df.to_csv(output,index=False)
        output.seek(0)

        return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition":"attachment; filename=predictions.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred while making predictions: {str(e)}")
