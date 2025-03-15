from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import time
import logging
import random

# Import authentication utilities
from api.routes.auth import get_current_active_user, User

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

# Models
class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"

class Position(BaseModel):
    symbol: str
    quantity: float
    average_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    asset_class: AssetClass
    market_value: float
    cost_basis: float

class PortfolioSummary(BaseModel):
    total_value: float
    cash_balance: float
    buying_power: float
    margin_used: float
    margin_available: float
    day_pnl: float
    day_pnl_percent: float
    total_pnl: float
    total_pnl_percent: float

class Portfolio(BaseModel):
    positions: List[Position]
    summary: PortfolioSummary

class AllocationTarget(BaseModel):
    symbol: str
    target_percentage: float
    asset_class: AssetClass

class AllocationStrategy(BaseModel):
    name: str
    description: str
    targets: List[AllocationTarget]
    risk_score: float  # 1-10, 1 being lowest risk
    expected_return: float  # Annual percentage
    expected_volatility: float  # Annual percentage

class PortfolioAnalysis(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    r_squared: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk
    correlations: Dict[str, Dict[str, float]]

# Mock portfolio data (in production, this would come from a database)
def generate_mock_portfolio(user_id: str) -> Portfolio:
    """Generate mock portfolio data for testing"""
    # Generate random positions
    positions = []
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC", "ETH", "EUR/USD", "GBP/JPY", "XAU/USD"]
    asset_classes = [AssetClass.STOCK, AssetClass.STOCK, AssetClass.STOCK, AssetClass.STOCK, AssetClass.STOCK,
                     AssetClass.CRYPTO, AssetClass.CRYPTO, AssetClass.FOREX, AssetClass.FOREX, AssetClass.COMMODITY]
    
    for i, symbol in enumerate(symbols):
        # Only include some positions randomly
        if random.random() < 0.7:
            # Set base price based on symbol
            base_price = sum(ord(c) for c in symbol) % 1000 + 10
            quantity = random.uniform(1, 100)
            entry_price = base_price * 0.9  # Assume bought at 10% lower price
            current_price = base_price
            
            positions.append(Position(
                symbol=symbol,
                quantity=quantity,
                average_entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=(current_price - entry_price) * quantity,
                realized_pnl=random.uniform(-1000, 5000),
                asset_class=asset_classes[i],
                market_value=current_price * quantity,
                cost_basis=entry_price * quantity
            ))
    
    # Calculate portfolio summary
    total_value = sum(position.market_value for position in positions)
    cash_balance = 100000.0  # Placeholder
    margin_used = total_value * 0.5  # Assuming 50% margin requirement
    margin_available = cash_balance - margin_used
    buying_power = cash_balance + margin_available
    
    # Calculate P&L
    day_pnl = random.uniform(-5000, 10000)
    day_pnl_percent = day_pnl / (total_value + cash_balance) * 100
    total_pnl = sum(position.unrealized_pnl for position in positions) + sum(position.realized_pnl for position in positions)
    total_pnl_percent = total_pnl / (total_value + cash_balance - total_pnl) * 100
    
    summary = PortfolioSummary(
        total_value=total_value,
        cash_balance=cash_balance,
        buying_power=buying_power,
        margin_used=margin_used,
        margin_available=margin_available,
        day_pnl=day_pnl,
        day_pnl_percent=day_pnl_percent,
        total_pnl=total_pnl,
        total_pnl_percent=total_pnl_percent
    )
    
    return Portfolio(positions=positions, summary=summary)

def generate_mock_allocation_strategies() -> List[AllocationStrategy]:
    """Generate mock allocation strategies for testing"""
    strategies = []
    
    # Conservative strategy
    conservative = AllocationStrategy(
        name="Conservative",
        description="Low-risk strategy focused on capital preservation",
        targets=[
            AllocationTarget(symbol="Bond ETF", target_percentage=60, asset_class=AssetClass.STOCK),
            AllocationTarget(symbol="S&P 500 ETF", target_percentage=20, asset_class=AssetClass.STOCK),
            AllocationTarget(symbol="Gold ETF", target_percentage=10, asset_class=AssetClass.COMMODITY),
            AllocationTarget(symbol="Cash", target_percentage=10, asset_class=AssetClass.FOREX)
        ],
        risk_score=3,
        expected_return=5.0,
        expected_volatility=7.0
    )
    strategies.append(conservative)
    
    # Balanced strategy
    balanced = AllocationStrategy(
        name="Balanced",
        description="Moderate-risk strategy with balanced growth and income",
        targets=[
            AllocationTarget(symbol="S&P 500 ETF", target_percentage=40, asset_class=AssetClass.STOCK),
            AllocationTarget(symbol="Bond ETF", target_percentage=30, asset_class=AssetClass.STOCK),
            AllocationTarget(symbol="International ETF", target_percentage=20, asset_class=AssetClass.STOCK),
            AllocationTarget(symbol="BTC", target_percentage=5, asset_class=AssetClass.CRYPTO),
            AllocationTarget(symbol="Gold ETF", target_percentage=5, asset_class=AssetClass.COMMODITY)
        ],
        risk_score=5,
        expected_return=8.0,
        expected_volatility=12.0
    )
    strategies.append(balanced)
    
    # Aggressive strategy
    aggressive = AllocationStrategy(
        name="Aggressive Growth",
        description="High-risk strategy focused on capital appreciation",
        targets=[
            AllocationTarget(symbol="Tech ETF", target_percentage=40, asset_class=AssetClass.STOCK),
            AllocationTarget(symbol="Small Cap ETF", target_percentage=25, asset_class=AssetClass.STOCK),
            AllocationTarget(symbol="Emerging Markets ETF", target_percentage=15, asset_class=AssetClass.STOCK),
            AllocationTarget(symbol="BTC", target_percentage=10, asset_class=AssetClass.CRYPTO),
            AllocationTarget(symbol="ETH", target_percentage=10, asset_class=AssetClass.CRYPTO)
        ],
        risk_score=8,
        expected_return=15.0,
        expected_volatility=25.0
    )
    strategies.append(aggressive)
    
    return strategies

def generate_mock_portfolio_analysis(user_id: str) -> PortfolioAnalysis:
    """Generate mock portfolio analysis for testing"""
    # Generate random correlation matrix
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC", "ETH", "S&P 500"]
    correlations = {}
    
    for symbol1 in symbols:
        correlations[symbol1] = {}
        for symbol2 in symbols:
            if symbol1 == symbol2:
                correlations[symbol1][symbol2] = 1.0
            else:
                # Ensure symmetric correlation matrix
                if symbol2 in correlations and symbol1 in correlations[symbol2]:
                    correlations[symbol1][symbol2] = correlations[symbol2][symbol1]
                else:
                    # Generate random correlation between -1 and 1
                    correlations[symbol1][symbol2] = random.uniform(-0.5, 1.0)
    
    return PortfolioAnalysis(
        sharpe_ratio=random.uniform(0.5, 3.0),
        sortino_ratio=random.uniform(0.7, 4.0),
        max_drawdown=random.uniform(0.1, 0.5),
        beta=random.uniform(0.5, 1.5),
        alpha=random.uniform(-0.1, 0.2),
        r_squared=random.uniform(0.6, 0.95),
        var_95=random.uniform(0.05, 0.2),
        cvar_95=random.uniform(0.07, 0.25),
        correlations=correlations
    )

# Routes
@router.get("/summary", response_model=Portfolio)
async def get_portfolio_summary(
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate mock data (in production, fetch from database)
    portfolio = generate_mock_portfolio(current_user.username)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Portfolio summary request for user {current_user.username}, Latency: {latency_ms:.3f}ms")
    
    return portfolio

@router.get("/analysis", response_model=PortfolioAnalysis)
async def get_portfolio_analysis(
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate mock data (in production, calculate from real portfolio)
    analysis = generate_mock_portfolio_analysis(current_user.username)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Portfolio analysis request for user {current_user.username}, Latency: {latency_ms:.3f}ms")
    
    return analysis

@router.get("/allocation-strategies", response_model=List[AllocationStrategy])
async def get_allocation_strategies(
    risk_level: Optional[int] = Query(None, ge=1, le=10),
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate mock data
    strategies = generate_mock_allocation_strategies()
    
    # Filter by risk level if specified
    if risk_level is not None:
        strategies = [s for s in strategies if abs(s.risk_score - risk_level) <= 2]
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Allocation strategies request for user {current_user.username}, Latency: {latency_ms:.3f}ms")
    
    return strategies

@router.get("/positions", response_model=List[Position])
async def get_positions(
    asset_class: Optional[AssetClass] = None,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate mock data
    portfolio = generate_mock_portfolio(current_user.username)
    positions = portfolio.positions
    
    # Filter by asset class if specified
    if asset_class:
        positions = [p for p in positions if p.asset_class == asset_class]
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Positions request for user {current_user.username}, Latency: {latency_ms:.3f}ms")
    
    return positions
