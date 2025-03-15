from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
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
class TimeFrame(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"

class MarketDataSource(str, Enum):
    BINANCE = "binance"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    COINBASE = "coinbase"
    INTERACTIVE_BROKERS = "interactive_brokers"

class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketDepth(BaseModel):
    timestamp: datetime
    bids: List[Dict[str, float]]  # price, quantity
    asks: List[Dict[str, float]]  # price, quantity

class Ticker(BaseModel):
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime

class NewsItem(BaseModel):
    title: str
    source: str
    url: str
    published_at: datetime
    summary: str
    sentiment: Optional[float] = None  # -1.0 to 1.0, negative to positive

class MarketSentiment(BaseModel):
    symbol: str
    sentiment_score: float  # -1.0 to 1.0, negative to positive
    news_count: int
    social_media_count: int
    timestamp: datetime
    sources: List[str]

# Mock data generators (in production, these would fetch from real APIs)
def generate_mock_ohlcv(symbol: str, timeframe: TimeFrame, start_time: datetime, end_time: datetime) -> List[OHLCV]:
    """Generate mock OHLCV data for testing"""
    data = []
    current_time = start_time
    
    # Set initial price based on symbol (just for variety in the mock data)
    base_price = sum(ord(c) for c in symbol) % 1000 + 10
    
    # Get time delta based on timeframe
    if timeframe == TimeFrame.ONE_MINUTE:
        delta = timedelta(minutes=1)
    elif timeframe == TimeFrame.FIVE_MINUTES:
        delta = timedelta(minutes=5)
    elif timeframe == TimeFrame.FIFTEEN_MINUTES:
        delta = timedelta(minutes=15)
    elif timeframe == TimeFrame.ONE_HOUR:
        delta = timedelta(hours=1)
    elif timeframe == TimeFrame.FOUR_HOURS:
        delta = timedelta(hours=4)
    elif timeframe == TimeFrame.ONE_DAY:
        delta = timedelta(days=1)
    elif timeframe == TimeFrame.ONE_WEEK:
        delta = timedelta(weeks=1)
    else:  # ONE_MONTH
        delta = timedelta(days=30)
    
    # Generate data points
    current_price = base_price
    while current_time <= end_time:
        # Random price movement
        price_change = current_price * random.uniform(-0.01, 0.01)
        open_price = current_price
        close_price = current_price + price_change
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
        volume = random.uniform(1000, 10000)
        
        data.append(OHLCV(
            timestamp=current_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        ))
        
        current_price = close_price
        current_time += delta
    
    return data

def generate_mock_market_depth(symbol: str) -> MarketDepth:
    """Generate mock market depth data for testing"""
    # Set base price based on symbol
    base_price = sum(ord(c) for c in symbol) % 1000 + 10
    
    # Generate bids (buy orders) below the base price
    bids = []
    for i in range(10):
        price = base_price * (1 - 0.001 * (i + 1))
        quantity = random.uniform(1, 10) * (10 - i)
        bids.append({"price": price, "quantity": quantity})
    
    # Generate asks (sell orders) above the base price
    asks = []
    for i in range(10):
        price = base_price * (1 + 0.001 * (i + 1))
        quantity = random.uniform(1, 10) * (10 - i)
        asks.append({"price": price, "quantity": quantity})
    
    return MarketDepth(
        timestamp=datetime.utcnow(),
        bids=bids,
        asks=asks
    )

def generate_mock_ticker(symbol: str) -> Ticker:
    """Generate mock ticker data for testing"""
    # Set base price based on symbol
    base_price = sum(ord(c) for c in symbol) % 1000 + 10
    
    # Generate random price changes
    change_percent = random.uniform(-5, 5)
    change = base_price * change_percent / 100
    
    return Ticker(
        symbol=symbol,
        last_price=base_price,
        bid=base_price * 0.999,
        ask=base_price * 1.001,
        volume_24h=random.uniform(10000, 1000000),
        change_24h=change,
        change_percent_24h=change_percent,
        high_24h=base_price * (1 + random.uniform(0, 0.05)),
        low_24h=base_price * (1 - random.uniform(0, 0.05)),
        timestamp=datetime.utcnow()
    )

def generate_mock_news(symbol: str, count: int = 5) -> List[NewsItem]:
    """Generate mock news data for testing"""
    news = []
    sources = ["Bloomberg", "Reuters", "CNBC", "Wall Street Journal", "Financial Times"]
    
    for i in range(count):
        # Generate a random sentiment score between -1 and 1
        sentiment = random.uniform(-1, 1)
        
        # Generate a title based on the sentiment
        if sentiment > 0.3:
            title = f"{symbol} shows strong growth potential, analysts say"
        elif sentiment < -0.3:
            title = f"Investors concerned about {symbol}'s future prospects"
        else:
            title = f"{symbol} remains stable amid market fluctuations"
        
        news.append(NewsItem(
            title=title,
            source=random.choice(sources),
            url=f"https://example.com/news/{symbol.lower()}/{i}",
            published_at=datetime.utcnow() - timedelta(hours=random.randint(1, 24)),
            summary=f"This is a mock summary for {symbol} news item {i}.",
            sentiment=sentiment
        ))
    
    return news

def generate_mock_sentiment(symbol: str) -> MarketSentiment:
    """Generate mock market sentiment data for testing"""
    # Generate a random sentiment score between -1 and 1
    sentiment_score = random.uniform(-1, 1)
    
    return MarketSentiment(
        symbol=symbol,
        sentiment_score=sentiment_score,
        news_count=random.randint(5, 20),
        social_media_count=random.randint(50, 500),
        timestamp=datetime.utcnow(),
        sources=["News Analysis", "Twitter Sentiment", "Reddit Analysis", "StockTwits"]
    )

# Routes
@router.get("/historical/{symbol}", response_model=List[OHLCV])
async def get_historical_data(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.ONE_DAY,
    start_date: datetime = Query(default=None),
    end_date: datetime = Query(default=None),
    source: MarketDataSource = MarketDataSource.YAHOO_FINANCE,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        # Default to 30 days of data
        start_date = end_date - timedelta(days=30)
    
    # Generate mock data (in production, fetch from real APIs)
    data = generate_mock_ohlcv(symbol, timeframe, start_date, end_date)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Historical data request: {symbol}, {timeframe}, Latency: {latency_ms:.3f}ms")
    
    # Check if latency exceeds threshold
    if latency_ms > 10:
        logger.warning(f"High latency detected: {latency_ms:.3f}ms for historical data request")
    
    return data

@router.get("/depth/{symbol}", response_model=MarketDepth)
async def get_market_depth(
    symbol: str,
    source: MarketDataSource = MarketDataSource.BINANCE,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate mock data (in production, fetch from real APIs)
    data = generate_mock_market_depth(symbol)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Market depth request: {symbol}, Latency: {latency_ms:.3f}ms")
    
    # Check if latency exceeds threshold
    if latency_ms > 5:
        logger.warning(f"High latency detected: {latency_ms:.3f}ms for market depth request")
    
    return data

@router.get("/ticker/{symbol}", response_model=Ticker)
async def get_ticker(
    symbol: str,
    source: MarketDataSource = MarketDataSource.YAHOO_FINANCE,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate mock data (in production, fetch from real APIs)
    data = generate_mock_ticker(symbol)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Ticker request: {symbol}, Latency: {latency_ms:.3f}ms")
    
    # Check if latency exceeds threshold
    if latency_ms > 5:
        logger.warning(f"High latency detected: {latency_ms:.3f}ms for ticker request")
    
    return data

@router.get("/news/{symbol}", response_model=List[NewsItem])
async def get_news(
    symbol: str,
    count: int = 10,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate mock data (in production, fetch from real APIs)
    data = generate_mock_news(symbol, count)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"News request: {symbol}, Latency: {latency_ms:.3f}ms")
    
    return data

@router.get("/sentiment/{symbol}", response_model=MarketSentiment)
async def get_sentiment(
    symbol: str,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate mock data (in production, fetch from real APIs)
    data = generate_mock_sentiment(symbol)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Sentiment request: {symbol}, Latency: {latency_ms:.3f}ms")
    
    return data

@router.get("/multi-ticker", response_model=List[Ticker])
async def get_multiple_tickers(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    source: MarketDataSource = MarketDataSource.YAHOO_FINANCE,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(",")]
    
    # Generate mock data for each symbol
    data = [generate_mock_ticker(symbol) for symbol in symbol_list]
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Multi-ticker request: {symbols}, Latency: {latency_ms:.3f}ms")
    
    # Check if latency exceeds threshold
    if latency_ms > 10:
        logger.warning(f"High latency detected: {latency_ms:.3f}ms for multi-ticker request")
    
    return data
