from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import time
import logging
import random
import numpy as np

# Import authentication utilities
from api.routes.auth import get_current_active_user, User

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

# Models
class RiskModel(str, Enum):
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric"
    EXPECTED_SHORTFALL = "expected_shortfall"
    STRESS_TEST = "stress_test"
    MONTE_CARLO = "monte_carlo"
    GARCH = "garch"

class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    PORTFOLIO = "portfolio"

class RiskMetric(BaseModel):
    name: str
    value: float
    description: str
    confidence_level: Optional[float] = None
    time_horizon: Optional[str] = None

class RiskAssessment(BaseModel):
    id: str
    asset_id: str
    asset_class: AssetClass
    model: RiskModel
    metrics: List[RiskMetric]
    timestamp: datetime
    latency_ms: float

class StressTestScenario(str, Enum):
    MARKET_CRASH = "market_crash"
    INTEREST_RATE_SPIKE = "interest_rate_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    GEOPOLITICAL_CRISIS = "geopolitical_crisis"
    PANDEMIC = "pandemic"
    TECH_BUBBLE_BURST = "tech_bubble_burst"
    ENERGY_CRISIS = "energy_crisis"

class StressTestResult(BaseModel):
    id: str
    portfolio_id: str
    scenario: StressTestScenario
    potential_loss: float
    potential_loss_percentage: float
    affected_assets: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime

class VolatilityForecast(BaseModel):
    id: str
    asset_id: str
    asset_class: AssetClass
    current_volatility: float
    forecast_1d: float
    forecast_1w: float
    forecast_1m: float
    forecast_3m: float
    confidence_intervals: Dict[str, Dict[str, float]]
    model: str
    timestamp: datetime

class CorrelationMatrix(BaseModel):
    assets: List[str]
    matrix: List[List[float]]
    period: str
    timestamp: datetime

# Mock risk assessment functions
def calculate_mock_var(asset_id: str, confidence_level: float = 0.95, time_horizon: int = 1) -> float:
    """Calculate mock Value at Risk (VaR)"""
    # In a real system, this would use historical data and proper statistical methods
    # For this mock, we'll generate a random value based on the asset
    base_value = sum(ord(c) for c in asset_id) % 100
    
    # Higher confidence level means higher VaR
    confidence_factor = 1 + (confidence_level - 0.95) * 10
    
    # Longer time horizon means higher VaR (square root of time rule)
    time_factor = np.sqrt(time_horizon)
    
    # Generate a random VaR between 1% and 10% of the base value
    var_percentage = random.uniform(0.01, 0.1) * confidence_factor * time_factor
    
    return var_percentage

def calculate_mock_expected_shortfall(asset_id: str, confidence_level: float = 0.95, time_horizon: int = 1) -> float:
    """Calculate mock Expected Shortfall (ES) / Conditional VaR"""
    # ES is typically higher than VaR
    var = calculate_mock_var(asset_id, confidence_level, time_horizon)
    # ES is typically 20-40% higher than VaR
    es_factor = random.uniform(1.2, 1.4)
    
    return var * es_factor

def generate_mock_risk_assessment(asset_id: str, asset_class: AssetClass, model: RiskModel) -> RiskAssessment:
    """Generate mock risk assessment for testing"""
    # Start timer for latency measurement
    start_time = time.time()
    
    metrics = []
    
    # Generate different metrics based on the model
    if model == RiskModel.VAR_HISTORICAL or model == RiskModel.VAR_PARAMETRIC:
        # 1-day VaR at 95% confidence
        var_95_1d = calculate_mock_var(asset_id, 0.95, 1)
        metrics.append(RiskMetric(
            name="VaR (95%, 1-day)",
            value=var_95_1d,
            description="Value at Risk with 95% confidence over 1 day",
            confidence_level=0.95,
            time_horizon="1d"
        ))
        
        # 1-day VaR at 99% confidence
        var_99_1d = calculate_mock_var(asset_id, 0.99, 1)
        metrics.append(RiskMetric(
            name="VaR (99%, 1-day)",
            value=var_99_1d,
            description="Value at Risk with 99% confidence over 1 day",
            confidence_level=0.99,
            time_horizon="1d"
        ))
        
        # 10-day VaR at 99% confidence (for regulatory purposes)
        var_99_10d = calculate_mock_var(asset_id, 0.99, 10)
        metrics.append(RiskMetric(
            name="VaR (99%, 10-day)",
            value=var_99_10d,
            description="Value at Risk with 99% confidence over 10 days",
            confidence_level=0.99,
            time_horizon="10d"
        ))
    
    if model == RiskModel.EXPECTED_SHORTFALL:
        # 1-day ES at 97.5% confidence
        es_975_1d = calculate_mock_expected_shortfall(asset_id, 0.975, 1)
        metrics.append(RiskMetric(
            name="Expected Shortfall (97.5%, 1-day)",
            value=es_975_1d,
            description="Expected loss given that the loss exceeds the 97.5% VaR",
            confidence_level=0.975,
            time_horizon="1d"
        ))
        
        # 1-day ES at 99% confidence
        es_99_1d = calculate_mock_expected_shortfall(asset_id, 0.99, 1)
        metrics.append(RiskMetric(
            name="Expected Shortfall (99%, 1-day)",
            value=es_99_1d,
            description="Expected loss given that the loss exceeds the 99% VaR",
            confidence_level=0.99,
            time_horizon="1d"
        ))
    
    if model == RiskModel.STRESS_TEST:
        # Market crash scenario
        metrics.append(RiskMetric(
            name="Market Crash Impact",
            value=random.uniform(0.1, 0.5),
            description="Estimated loss in a severe market crash scenario",
            time_horizon="event"
        ))
        
        # Interest rate spike scenario
        metrics.append(RiskMetric(
            name="Interest Rate Spike Impact",
            value=random.uniform(0.05, 0.3),
            description="Estimated loss in a sudden interest rate spike scenario",
            time_horizon="event"
        ))
    
    if model == RiskModel.MONTE_CARLO:
        # Monte Carlo VaR
        mc_var_95 = calculate_mock_var(asset_id, 0.95, 1) * random.uniform(0.9, 1.1)
        metrics.append(RiskMetric(
            name="Monte Carlo VaR (95%, 1-day)",
            value=mc_var_95,
            description="VaR calculated using Monte Carlo simulation",
            confidence_level=0.95,
            time_horizon="1d"
        ))
        
        # Monte Carlo ES
        mc_es_95 = mc_var_95 * random.uniform(1.2, 1.4)
        metrics.append(RiskMetric(
            name="Monte Carlo ES (95%, 1-day)",
            value=mc_es_95,
            description="Expected Shortfall calculated using Monte Carlo simulation",
            confidence_level=0.95,
            time_horizon="1d"
        ))
    
    if model == RiskModel.GARCH:
        # Current volatility
        current_vol = random.uniform(0.01, 0.05)
        metrics.append(RiskMetric(
            name="Current Volatility",
            value=current_vol,
            description="Current daily volatility estimated using GARCH",
            time_horizon="1d"
        ))
        
        # Forecasted volatility
        forecast_vol = current_vol * random.uniform(0.8, 1.2)
        metrics.append(RiskMetric(
            name="Forecasted Volatility (1-week)",
            value=forecast_vol,
            description="Forecasted daily volatility for next week using GARCH",
            time_horizon="1w"
        ))
    
    # Add some general metrics for all models
    metrics.append(RiskMetric(
        name="Beta",
        value=random.uniform(0.5, 1.5),
        description="Sensitivity to market movements"
    ))
    
    metrics.append(RiskMetric(
        name="Sharpe Ratio",
        value=random.uniform(0.5, 3.0),
        description="Risk-adjusted return"
    ))
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    return RiskAssessment(
        id=f"risk-{int(time.time())}-{random.randint(1000, 9999)}",
        asset_id=asset_id,
        asset_class=asset_class,
        model=model,
        metrics=metrics,
        timestamp=datetime.utcnow(),
        latency_ms=latency_ms
    )

def generate_mock_stress_test(portfolio_id: str, scenario: StressTestScenario) -> StressTestResult:
    """Generate mock stress test results for testing"""
    # Generate potential loss based on scenario
    if scenario == StressTestScenario.MARKET_CRASH:
        potential_loss_percentage = random.uniform(0.25, 0.5)
        affected_assets = [
            {"asset_id": "SPY", "loss_percentage": random.uniform(0.3, 0.6)},
            {"asset_id": "QQQ", "loss_percentage": random.uniform(0.35, 0.65)},
            {"asset_id": "IWM", "loss_percentage": random.uniform(0.4, 0.7)}
        ]
        recommendations = [
            "Increase allocation to defensive sectors",
            "Consider protective put options",
            "Maintain higher cash reserves",
            "Implement tail risk hedging strategies"
        ]
    elif scenario == StressTestScenario.INTEREST_RATE_SPIKE:
        potential_loss_percentage = random.uniform(0.1, 0.3)
        affected_assets = [
            {"asset_id": "TLT", "loss_percentage": random.uniform(0.15, 0.35)},
            {"asset_id": "AGG", "loss_percentage": random.uniform(0.1, 0.25)},
            {"asset_id": "REIT ETFs", "loss_percentage": random.uniform(0.2, 0.4)}
        ]
        recommendations = [
            "Reduce duration in fixed income holdings",
            "Increase allocation to floating rate securities",
            "Consider interest rate hedges",
            "Shift to value stocks from growth stocks"
        ]
    elif scenario == StressTestScenario.LIQUIDITY_CRISIS:
        potential_loss_percentage = random.uniform(0.2, 0.4)
        affected_assets = [
            {"asset_id": "High Yield Bonds", "loss_percentage": random.uniform(0.25, 0.45)},
            {"asset_id": "Emerging Market Debt", "loss_percentage": random.uniform(0.3, 0.5)},
            {"asset_id": "Small Cap Stocks", "loss_percentage": random.uniform(0.25, 0.45)}
        ]
        recommendations = [
            "Increase allocation to highly liquid assets",
            "Reduce exposure to illiquid alternative investments",
            "Establish backup liquidity facilities",
            "Implement staged liquidation protocols"
        ]
    else:  # Generic scenario
        potential_loss_percentage = random.uniform(0.15, 0.35)
        affected_assets = [
            {"asset_id": "Equities", "loss_percentage": random.uniform(0.2, 0.4)},
            {"asset_id": "Corporate Bonds", "loss_percentage": random.uniform(0.15, 0.3)},
            {"asset_id": "Commodities", "loss_percentage": random.uniform(0.1, 0.3)}
        ]
        recommendations = [
            "Diversify across asset classes",
            "Implement downside protection strategies",
            "Maintain adequate liquidity reserves",
            "Consider alternative investments with low correlation"
        ]
    
    # Assume a portfolio value of $1,000,000 for calculating absolute loss
    portfolio_value = 1000000
    potential_loss = portfolio_value * potential_loss_percentage
    
    return StressTestResult(
        id=f"stress-{int(time.time())}-{random.randint(1000, 9999)}",
        portfolio_id=portfolio_id,
        scenario=scenario,
        potential_loss=potential_loss,
        potential_loss_percentage=potential_loss_percentage,
        affected_assets=affected_assets,
        recommendations=recommendations,
        timestamp=datetime.utcnow()
    )

def generate_mock_volatility_forecast(asset_id: str, asset_class: AssetClass) -> VolatilityForecast:
    """Generate mock volatility forecast for testing"""
    # Generate current volatility based on asset class
    if asset_class == AssetClass.STOCK:
        current_volatility = random.uniform(0.01, 0.03)
    elif asset_class == AssetClass.CRYPTO:
        current_volatility = random.uniform(0.03, 0.08)
    elif asset_class == AssetClass.FOREX:
        current_volatility = random.uniform(0.005, 0.015)
    elif asset_class == AssetClass.COMMODITY:
        current_volatility = random.uniform(0.015, 0.04)
    else:  # Portfolio
        current_volatility = random.uniform(0.008, 0.025)
    
    # Generate forecasts with some randomness but generally increasing with time
    forecast_1d = current_volatility * random.uniform(0.95, 1.05)
    forecast_1w = current_volatility * random.uniform(1.0, 1.1)
    forecast_1m = current_volatility * random.uniform(1.05, 1.2)
    forecast_3m = current_volatility * random.uniform(1.1, 1.3)
    
    # Generate confidence intervals
    confidence_intervals = {
        "1d": {
            "lower_95": forecast_1d * 0.8,
            "upper_95": forecast_1d * 1.2
        },
        "1w": {
            "lower_95": forecast_1w * 0.75,
            "upper_95": forecast_1w * 1.25
        },
        "1m": {
            "lower_95": forecast_1m * 0.7,
            "upper_95": forecast_1m * 1.3
        },
        "3m": {
            "lower_95": forecast_3m * 0.6,
            "upper_95": forecast_3m * 1.4
        }
    }
    
    return VolatilityForecast(
        id=f"vol-{int(time.time())}-{random.randint(1000, 9999)}",
        asset_id=asset_id,
        asset_class=asset_class,
        current_volatility=current_volatility,
        forecast_1d=forecast_1d,
        forecast_1w=forecast_1w,
        forecast_1m=forecast_1m,
        forecast_3m=forecast_3m,
        confidence_intervals=confidence_intervals,
        model="GARCH(1,1)",
        timestamp=datetime.utcnow()
    )

def generate_mock_correlation_matrix(assets: List[str], period: str) -> CorrelationMatrix:
    """Generate mock correlation matrix for testing"""
    n = len(assets)
    
    # Initialize with identity matrix
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Fill in correlations (ensuring symmetry)
    for i in range(n):
        for j in range(i+1, n):
            # Generate random correlation between -0.2 and 0.9
            # Most financial assets have positive correlation
            corr = random.uniform(-0.2, 0.9)
            matrix[i][j] = corr
            matrix[j][i] = corr  # Ensure symmetry
    
    return CorrelationMatrix(
        assets=assets,
        matrix=matrix,
        period=period,
        timestamp=datetime.utcnow()
    )

# Routes
@router.get("/assessment/{asset_id}", response_model=RiskAssessment)
async def get_risk_assessment(
    asset_id: str,
    asset_class: AssetClass,
    model: RiskModel = RiskModel.VAR_HISTORICAL,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate risk assessment
    assessment = generate_mock_risk_assessment(asset_id, asset_class, model)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Risk assessment request: Asset={asset_id}, Model={model}, Latency: {latency_ms:.3f}ms")
    
    # Check if latency exceeds threshold
    if latency_ms > 5:
        logger.warning(f"High latency detected: {latency_ms:.3f}ms for risk assessment")
    
    return assessment

@router.get("/stress-test/{portfolio_id}", response_model=StressTestResult)
async def get_stress_test(
    portfolio_id: str,
    scenario: StressTestScenario = StressTestScenario.MARKET_CRASH,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate stress test results
    result = generate_mock_stress_test(portfolio_id, scenario)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Stress test request: Portfolio={portfolio_id}, Scenario={scenario}, Latency: {latency_ms:.3f}ms")
    
    return result

@router.get("/volatility/{asset_id}", response_model=VolatilityForecast)
async def get_volatility_forecast(
    asset_id: str,
    asset_class: AssetClass,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate volatility forecast
    forecast = generate_mock_volatility_forecast(asset_id, asset_class)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Volatility forecast request: Asset={asset_id}, Latency: {latency_ms:.3f}ms")
    
    return forecast

@router.get("/correlation", response_model=CorrelationMatrix)
async def get_correlation_matrix(
    assets: str = Query(..., description="Comma-separated list of asset IDs"),
    period: str = "1y",
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Parse assets
    asset_list = [a.strip() for a in assets.split(",")]
    
    # Generate correlation matrix
    matrix = generate_mock_correlation_matrix(asset_list, period)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Correlation matrix request: Assets={assets}, Period={period}, Latency: {latency_ms:.3f}ms")
    
    return matrix
