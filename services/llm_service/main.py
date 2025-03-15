import os
import time
import logging
import random
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

import numpy as np
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_service")

# Initialize FastAPI app
app = FastAPI(title="FinTech LLM Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class SentimentAnalysisRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None

class SentimentAnalysisResponse(BaseModel):
    sentiment_score: float  # -1.0 (very negative) to 1.0 (very positive)
    confidence: float  # 0.0 to 1.0
    entities: Dict[str, float]  # Entity -> sentiment score
    summary: str
    processing_time_ms: float

class DocumentSummaryRequest(BaseModel):
    document: str
    max_length: Optional[int] = 500
    focus_areas: Optional[List[str]] = None

class DocumentSummaryResponse(BaseModel):
    summary: str
    key_points: List[str]
    entities: Dict[str, str]  # Entity -> description/relevance
    processing_time_ms: float

class TradingAdviceRequest(BaseModel):
    portfolio: Dict[str, float]  # Symbol -> allocation percentage
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0)  # 0.0 (low risk) to 1.0 (high risk)
    investment_horizon: str  # short_term, medium_term, long_term
    market_conditions: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

class TradingAdviceResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    rationale: str
    risk_assessment: Dict[str, Any]
    processing_time_ms: float

class FraudDetectionRequest(BaseModel):
    transaction_data: Dict[str, Any]
    user_history: Optional[Dict[str, Any]] = None
    additional_context: Optional[Dict[str, Any]] = None

class FraudDetectionResponse(BaseModel):
    fraud_score: float  # 0.0 (not fraud) to 1.0 (definitely fraud)
    confidence: float  # 0.0 to 1.0
    risk_factors: List[Dict[str, Any]]
    recommendations: List[str]
    processing_time_ms: float

# Mock LLM functions
def mock_sentiment_analysis(text: str, context: Optional[Dict[str, Any]] = None) -> SentimentAnalysisResponse:
    """Mock sentiment analysis function"""
    start_time = time.time()
    
    # Extract potential entities (simple implementation for mock)
    words = text.lower().split()
    potential_entities = [
        "market", "stock", "bond", "equity", "investment", "growth", "recession", 
        "inflation", "interest", "rate", "fed", "earnings", "revenue", "profit", 
        "loss", "dividend", "volatility", "bull", "bear", "rally", "correction"
    ]
    
    entities = {}
    for entity in potential_entities:
        if entity in words:
            # Generate a sentiment score for this entity
            entities[entity] = round(random.uniform(-1.0, 1.0), 2)
    
    # Generate overall sentiment based on words in the text
    positive_words = ["increase", "growth", "profit", "gain", "positive", "bull", "rally", "up", "rise", "good", "strong"]
    negative_words = ["decrease", "recession", "loss", "negative", "bear", "correction", "down", "fall", "bad", "weak"]
    
    positive_count = sum(1 for word in positive_words if word in words)
    negative_count = sum(1 for word in negative_words if word in words)
    
    # Calculate sentiment score
    if positive_count + negative_count > 0:
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
    else:
        sentiment_score = 0.0
    
    # Add some randomness
    sentiment_score = max(-1.0, min(1.0, sentiment_score + random.uniform(-0.2, 0.2)))
    
    # Generate confidence
    confidence = random.uniform(0.7, 0.95)
    
    # Generate summary
    if sentiment_score > 0.3:
        summary = "The text expresses a positive sentiment towards the financial markets or assets discussed."
    elif sentiment_score < -0.3:
        summary = "The text expresses a negative sentiment towards the financial markets or assets discussed."
    else:
        summary = "The text expresses a neutral or mixed sentiment towards the financial markets or assets discussed."
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return SentimentAnalysisResponse(
        sentiment_score=round(sentiment_score, 2),
        confidence=round(confidence, 2),
        entities=entities,
        summary=summary,
        processing_time_ms=round(processing_time_ms, 2)
    )

def mock_document_summary(document: str, max_length: int = 500, focus_areas: Optional[List[str]] = None) -> DocumentSummaryResponse:
    """Mock document summary function"""
    start_time = time.time()
    
    # Generate a mock summary (in a real system, this would use an actual LLM)
    words = document.split()
    total_words = len(words)
    
    # Calculate summary length (about 20% of original, but respect max_length)
    summary_length = min(max_length, max(50, total_words // 5))
    
    if total_words <= summary_length:
        summary = document
    else:
        # Take first 40% and last 60% of the summary length
        first_part_length = summary_length * 2 // 5
        last_part_length = summary_length - first_part_length
        
        first_part = " ".join(words[:first_part_length])
        last_part = " ".join(words[-(last_part_length):])
        
        summary = f"{first_part} [...] {last_part}"
    
    # Generate key points
    num_points = random.randint(3, 6)
    key_points = []
    
    for i in range(num_points):
        start_idx = random.randint(0, max(0, total_words - 10))
        point_length = random.randint(5, 10)
        point = " ".join(words[start_idx:min(start_idx + point_length, total_words)])
        key_points.append(f"Point {i+1}: {point}")
    
    # Generate entities
    entities = {}
    potential_entities = [
        "market", "stock", "bond", "equity", "investment", "growth", "recession", 
        "inflation", "interest rate", "federal reserve", "earnings", "revenue", 
        "profit margin", "dividend yield", "volatility index", "bull market", 
        "bear market", "market rally", "correction", "sector rotation"
    ]
    
    num_entities = random.randint(3, 8)
    selected_entities = random.sample(potential_entities, min(num_entities, len(potential_entities)))
    
    for entity in selected_entities:
        entities[entity] = f"Mentioned in context of {'positive' if random.random() > 0.5 else 'negative'} market outlook"
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return DocumentSummaryResponse(
        summary=summary,
        key_points=key_points,
        entities=entities,
        processing_time_ms=round(processing_time_ms, 2)
    )

def mock_trading_advice(
    portfolio: Dict[str, float],
    risk_tolerance: float,
    investment_horizon: str,
    market_conditions: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None
) -> TradingAdviceResponse:
    """Mock trading advice function"""
    start_time = time.time()
    
    # Generate recommendations based on risk tolerance and investment horizon
    recommendations = []
    
    # Define asset classes and their risk profiles
    asset_classes = {
        "US Large Cap Equities": {"risk": 0.6, "tickers": ["SPY", "VOO", "VTI", "QQQ"]},
        "US Small Cap Equities": {"risk": 0.8, "tickers": ["IWM", "VB", "IJR"]},
        "International Developed Equities": {"risk": 0.7, "tickers": ["VEA", "EFA", "IEFA"]},
        "Emerging Market Equities": {"risk": 0.9, "tickers": ["VWO", "IEMG", "EEM"]},
        "US Government Bonds": {"risk": 0.2, "tickers": ["IEF", "TLT", "BND"]},
        "US Corporate Bonds": {"risk": 0.3, "tickers": ["LQD", "VCIT", "VCSH"]},
        "High Yield Bonds": {"risk": 0.5, "tickers": ["HYG", "JNK", "USHY"]},
        "REITs": {"risk": 0.7, "tickers": ["VNQ", "IYR", "SCHH"]},
        "Commodities": {"risk": 0.8, "tickers": ["GLD", "IAU", "GSG", "DBC"]},
        "Cash Equivalents": {"risk": 0.1, "tickers": ["SHV", "BIL", "SGOV"]}
    }
    
    # Adjust allocations based on risk tolerance and investment horizon
    if investment_horizon == "short_term":
        horizon_factor = 0.3
    elif investment_horizon == "medium_term":
        horizon_factor = 0.6
    else:  # long_term
        horizon_factor = 0.9
    
    # Combined risk factor (0.0 to 1.0)
    combined_risk = (risk_tolerance * 0.7) + (horizon_factor * 0.3)
    
    # Generate allocations
    allocations = {}
    
    # Stocks allocation increases with risk tolerance and investment horizon
    stocks_allocation = combined_risk * 0.8 + 0.2  # 20% to 100%
    
    # Bonds allocation is inverse to stocks
    bonds_allocation = 1.0 - stocks_allocation
    
    # Further break down stocks and bonds
    if stocks_allocation > 0:
        allocations["US Large Cap Equities"] = stocks_allocation * random.uniform(0.3, 0.5)
        allocations["US Small Cap Equities"] = stocks_allocation * random.uniform(0.05, 0.2)
        allocations["International Developed Equities"] = stocks_allocation * random.uniform(0.1, 0.3)
        allocations["Emerging Market Equities"] = stocks_allocation * random.uniform(0.05, 0.15)
        allocations["REITs"] = stocks_allocation * random.uniform(0.0, 0.1)
    
    if bonds_allocation > 0:
        allocations["US Government Bonds"] = bonds_allocation * random.uniform(0.4, 0.6)
        allocations["US Corporate Bonds"] = bonds_allocation * random.uniform(0.2, 0.4)
        allocations["High Yield Bonds"] = bonds_allocation * random.uniform(0.0, 0.2)
    
    # Add commodities and cash based on risk profile
    allocations["Commodities"] = combined_risk * random.uniform(0.0, 0.1)
    allocations["Cash Equivalents"] = (1.0 - combined_risk) * random.uniform(0.05, 0.2)
    
    # Normalize allocations to sum to 100%
    total_allocation = sum(allocations.values())
    for asset_class in allocations:
        allocations[asset_class] = allocations[asset_class] / total_allocation
    
    # Generate specific ticker recommendations
    for asset_class, allocation in allocations.items():
        if allocation >= 0.05:  # Only include if allocation is at least 5%
            tickers = asset_classes[asset_class]["tickers"]
            selected_ticker = random.choice(tickers)
            
            recommendations.append({
                "asset_class": asset_class,
                "ticker": selected_ticker,
                "allocation": round(allocation * 100, 1),
                "action": "buy" if allocation > 0.1 else "hold",
                "rationale": f"Allocated {round(allocation * 100, 1)}% to {asset_class} via {selected_ticker} based on {investment_horizon} horizon and risk tolerance."
            })
    
    # Sort recommendations by allocation (descending)
    recommendations.sort(key=lambda x: x["allocation"], reverse=True)
    
    # Generate rationale
    if combined_risk < 0.3:
        rationale = "Conservative portfolio focused on capital preservation with significant bond allocation and minimal exposure to high-risk assets."
    elif combined_risk < 0.6:
        rationale = "Balanced portfolio with moderate risk, diversified across stocks and bonds to provide growth potential while managing downside risk."
    else:
        rationale = "Growth-oriented portfolio with significant equity exposure to maximize long-term returns, suitable for investors with high risk tolerance."
    
    # Generate risk assessment
    risk_assessment = {
        "overall_risk_score": round(combined_risk * 10, 1),  # 0-10 scale
        "volatility_estimate": f"{round(combined_risk * 15, 1)}%",
        "drawdown_potential": f"{round(combined_risk * 35, 1)}%",
        "sharpe_ratio_estimate": round(1.0 + random.uniform(-0.5, 0.5), 2),
        "risk_factors": [
            "Market risk",
            "Interest rate risk",
            "Inflation risk",
            "Liquidity risk"
        ]
    }
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return TradingAdviceResponse(
        recommendations=recommendations,
        rationale=rationale,
        risk_assessment=risk_assessment,
        processing_time_ms=round(processing_time_ms, 2)
    )

def mock_fraud_detection(
    transaction_data: Dict[str, Any],
    user_history: Optional[Dict[str, Any]] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> FraudDetectionResponse:
    """Mock fraud detection function"""
    start_time = time.time()
    
    # Extract transaction details
    amount = transaction_data.get("amount", 0)
    transaction_type = transaction_data.get("type", "unknown")
    location = transaction_data.get("location", "unknown")
    time_of_day = transaction_data.get("time_of_day", "unknown")
    device = transaction_data.get("device", "unknown")
    
    # Initialize fraud score
    base_fraud_score = 0.0
    risk_factors = []
    
    # Check for high-risk indicators
    
    # 1. Unusual transaction amount
    if amount > 10000:
        base_fraud_score += 0.3
        risk_factors.append({
            "factor": "Large transaction amount",
            "severity": "high",
            "description": f"Transaction amount (${amount}) is unusually large"
        })
    elif amount > 5000:
        base_fraud_score += 0.15
        risk_factors.append({
            "factor": "Moderate transaction amount",
            "severity": "medium",
            "description": f"Transaction amount (${amount}) is moderately large"
        })
    
    # 2. Unusual location
    if user_history and "usual_locations" in user_history:
        usual_locations = user_history["usual_locations"]
        if location not in usual_locations:
            base_fraud_score += 0.25
            risk_factors.append({
                "factor": "Unusual transaction location",
                "severity": "high",
                "description": f"Transaction location ({location}) differs from usual patterns"
            })
    
    # 3. Unusual time of day
    if time_of_day in ["late_night", "early_morning"]:
        base_fraud_score += 0.1
        risk_factors.append({
            "factor": "Unusual transaction time",
            "severity": "low",
            "description": f"Transaction occurred during unusual hours ({time_of_day})"
        })
    
    # 4. Unusual device
    if user_history and "usual_devices" in user_history:
        usual_devices = user_history["usual_devices"]
        if device not in usual_devices:
            base_fraud_score += 0.2
            risk_factors.append({
                "factor": "Unusual device",
                "severity": "medium",
                "description": f"Transaction made from unfamiliar device ({device})"
            })
    
    # 5. High-risk transaction type
    high_risk_types = ["wire_transfer", "crypto_purchase", "gift_card_purchase"]
    if transaction_type in high_risk_types:
        base_fraud_score += 0.15
        risk_factors.append({
            "factor": "High-risk transaction type",
            "severity": "medium",
            "description": f"Transaction type ({transaction_type}) is associated with higher fraud risk"
        })
    
    # 6. Rapid succession of transactions
    if user_history and "recent_transactions" in user_history:
        recent_transactions = user_history["recent_transactions"]
        if len(recent_transactions) > 5:
            base_fraud_score += 0.15
            risk_factors.append({
                "factor": "Multiple recent transactions",
                "severity": "medium",
                "description": "Unusual number of transactions in a short time period"
            })
    
    # Add some randomness to the fraud score
    fraud_score = min(1.0, max(0.0, base_fraud_score + random.uniform(-0.1, 0.1)))
    
    # Generate confidence based on available data
    confidence_factors = 0
    if user_history:
        confidence_factors += 1
    if additional_context:
        confidence_factors += 1
    
    base_confidence = 0.7  # Start with 70% confidence
    confidence_boost = confidence_factors * 0.1  # Each factor adds 10%
    confidence = min(0.95, base_confidence + confidence_boost)
    
    # Generate recommendations based on fraud score
    recommendations = []
    
    if fraud_score > 0.7:
        recommendations = [
            "Block transaction immediately",
            "Contact customer via registered phone number",
            "Request additional verification",
            "Flag account for enhanced monitoring"
        ]
    elif fraud_score > 0.4:
        recommendations = [
            "Request additional authentication",
            "Send verification code to registered phone number",
            "Apply transaction velocity limits",
            "Monitor account for next 24 hours"
        ]
    else:
        recommendations = [
            "Process transaction normally",
            "Apply standard monitoring protocols"
        ]
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return FraudDetectionResponse(
        fraud_score=round(fraud_score, 2),
        confidence=round(confidence, 2),
        risk_factors=risk_factors,
        recommendations=recommendations,
        processing_time_ms=round(processing_time_ms, 2)
    )

# Routes
@app.post("/sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment of financial text"""
    try:
        # Start timer for latency measurement
        start_time = time.time()
        
        # Process the request
        result = mock_sentiment_analysis(request.text, request.context)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log the request
        logger.info(f"Sentiment analysis request processed in {latency_ms:.3f}ms")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=DocumentSummaryResponse)
async def summarize_document(request: DocumentSummaryRequest):
    """Summarize financial document"""
    try:
        # Start timer for latency measurement
        start_time = time.time()
        
        # Process the request
        result = mock_document_summary(request.document, request.max_length, request.focus_areas)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log the request
        logger.info(f"Document summary request processed in {latency_ms:.3f}ms")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in document summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading-advice", response_model=TradingAdviceResponse)
async def get_trading_advice(request: TradingAdviceRequest):
    """Get AI-powered trading advice"""
    try:
        # Start timer for latency measurement
        start_time = time.time()
        
        # Process the request
        result = mock_trading_advice(
            request.portfolio,
            request.risk_tolerance,
            request.investment_horizon,
            request.market_conditions,
            request.constraints
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log the request
        logger.info(f"Trading advice request processed in {latency_ms:.3f}ms")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in trading advice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fraud-detection", response_model=FraudDetectionResponse)
async def detect_fraud(request: FraudDetectionRequest):
    """Detect potential fraud in financial transactions"""
    try:
        # Start timer for latency measurement
        start_time = time.time()
        
        # Process the request
        result = mock_fraud_detection(
            request.transaction_data,
            request.user_history,
            request.additional_context
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log the request
        logger.info(f"Fraud detection request processed in {latency_ms:.3f}ms")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "models_loaded": ["sentiment", "summarization", "trading_advice", "fraud_detection"]
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
