from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pendulum import DoublePendulum

app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/simulate")
def simulate_pendulum(theta1: float = 1.0, theta2: float = 0.5, T: float = 10.0, dt: float = 0.01):
    """
    Simulate double pendulum and return time series data.
    """
    pendulum = DoublePendulum(theta1=theta1, theta2=theta2, T=T, dt=dt)
    df = pendulum.generateTimeData()
    
    # Convert to dict for JSON
    data = {
        "t": df["t"].tolist(),
        "theta1": df["theta1"].tolist(),
        "theta2": df["theta2"].tolist(),
        "theta1_dot": df["theta1_dot"].tolist(),
        "theta2_dot": df["theta2_dot"].tolist(),
        "params": {
            "m1": df["m1"].iloc[0],
            "m2": df["m2"].iloc[0],
            "l1": df["l1"].iloc[0],
            "l2": df["l2"].iloc[0],
            "g": df["g"].iloc[0],
        }
    }
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)