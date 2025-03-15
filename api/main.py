from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create metrics
REQUEST_COUNT = Counter("api_requests_total", "Total count of requests by method and endpoint", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"])

# Create FastAPI app
app = FastAPI(
    title="FinTech LLM Platform API",
    description="High-Frequency, AI-Powered, Risk-Aware FinTech LLM System",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request timing and metrics
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    method = request.method
    path = request.url.path
    
    # Start timer
    start_time = time.time()
    
    # Process request
    try:
        response = await call_next(request)
        
        # Record metrics
        REQUEST_COUNT.labels(method=method, endpoint=path).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(time.time() - start_time)
        
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        # Record metrics for failed requests
        REQUEST_COUNT.labels(method=method, endpoint=path).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(time.time() - start_time)
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest())

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "FinTech LLM Platform API",
        "version": "1.0.0",
        "status": "operational"
    }

# Import and include routers
from api.routes import trading, market_data, portfolio, auth, llm, risk, compliance

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(trading.router, prefix="/trading", tags=["Trading"])
app.include_router(market_data.router, prefix="/market-data", tags=["Market Data"])
app.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])
app.include_router(llm.router, prefix="/llm", tags=["LLM Services"])
app.include_router(risk.router, prefix="/risk", tags=["Risk Assessment"])
app.include_router(compliance.router, prefix="/compliance", tags=["Compliance"])

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
