import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import aiohttp
import websockets
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_ingestion")

# Initialize FastAPI app
app = FastAPI(title="FinTech Data Ingestion Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class MarketDataPoint(BaseModel):
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str

class OrderBookEntry(BaseModel):
    price: float
    quantity: float
    num_orders: Optional[int] = None

class OrderBook(BaseModel):
    symbol: str
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    timestamp: datetime
    source: str

class NewsItem(BaseModel):
    id: str
    title: str
    summary: str
    url: Optional[str] = None
    source: str
    symbols: List[str]
    sentiment: Optional[float] = None
    timestamp: datetime

# In-memory data stores (in production, use Redis or similar)
market_data_cache = {}  # symbol -> latest data
order_book_cache = {}   # symbol -> order book
news_cache = []         # list of recent news items
connected_clients = {}  # client_id -> websocket connection

# Mock data generators
async def generate_mock_market_data(symbol: str) -> MarketDataPoint:
    """Generate realistic mock market data for testing"""
    # Get the last price or initialize
    last_data = market_data_cache.get(symbol)
    
    if last_data:
        # Generate realistic price movement
        price_change_pct = random.normalvariate(0, 0.001)  # Mean 0, std 0.1%
        new_price = last_data.price * (1 + price_change_pct)
        
        # Generate realistic volume
        volume_factor = random.uniform(0.5, 2.0)
        new_volume = last_data.volume * volume_factor
        
        # Generate bid/ask spread
        spread = new_price * random.uniform(0.0001, 0.002)  # 0.01% to 0.2% spread
        bid = new_price - spread / 2
        ask = new_price + spread / 2
    else:
        # Initialize with reasonable values based on symbol
        # Use the sum of ASCII values to generate different but consistent prices per symbol
        base_value = sum(ord(c) for c in symbol) % 1000
        new_price = base_value + random.uniform(-base_value * 0.01, base_value * 0.01)
        new_volume = random.uniform(100, 10000)
        
        # Generate bid/ask spread
        spread = new_price * random.uniform(0.0001, 0.002)  # 0.01% to 0.2% spread
        bid = new_price - spread / 2
        ask = new_price + spread / 2
    
    # Create data point
    data_point = MarketDataPoint(
        symbol=symbol,
        price=round(new_price, 4),
        volume=round(new_volume, 2),
        timestamp=datetime.utcnow(),
        bid=round(bid, 4),
        ask=round(ask, 4),
        source="mock_data_generator"
    )
    
    # Update cache
    market_data_cache[symbol] = data_point
    
    return data_point

async def generate_mock_order_book(symbol: str) -> OrderBook:
    """Generate realistic mock order book for testing"""
    # Get the last market data or generate new
    last_data = market_data_cache.get(symbol)
    if not last_data:
        last_data = await generate_mock_market_data(symbol)
    
    mid_price = last_data.price
    
    # Generate bids (buy orders) - lower than mid price
    bids = []
    current_bid = mid_price * 0.998  # Start 0.2% below mid price
    for i in range(10):  # 10 price levels
        price = round(current_bid * (1 - i * 0.001), 4)  # Each level is 0.1% lower
        quantity = round(random.uniform(10, 1000) * (1 + i * 0.5), 2)  # Higher quantity at lower prices
        num_orders = random.randint(1, 20)
        bids.append(OrderBookEntry(price=price, quantity=quantity, num_orders=num_orders))
    
    # Generate asks (sell orders) - higher than mid price
    asks = []
    current_ask = mid_price * 1.002  # Start 0.2% above mid price
    for i in range(10):  # 10 price levels
        price = round(current_ask * (1 + i * 0.001), 4)  # Each level is 0.1% higher
        quantity = round(random.uniform(10, 1000) * (1 + i * 0.5), 2)  # Higher quantity at higher prices
        num_orders = random.randint(1, 20)
        asks.append(OrderBookEntry(price=price, quantity=quantity, num_orders=num_orders))
    
    # Create order book
    order_book = OrderBook(
        symbol=symbol,
        bids=bids,
        asks=asks,
        timestamp=datetime.utcnow(),
        source="mock_data_generator"
    )
    
    # Update cache
    order_book_cache[symbol] = order_book
    
    return order_book

async def generate_mock_news() -> NewsItem:
    """Generate mock financial news for testing"""
    # List of mock news templates
    news_templates = [
        {
            "title": "{company} Reports {adj} Q{quarter} Earnings",
            "summary": "{company} reported {adj} than expected Q{quarter} earnings, with EPS of ${eps} vs ${exp_eps} expected. Revenue came in at ${revenue}B, {comp} the ${exp_revenue}B consensus estimate.",
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "GS", "MS"],
            "sentiment_range": (-0.8, 0.8)
        },
        {
            "title": "Fed {action} Interest Rates by {bps} Basis Points",
            "summary": "The Federal Reserve {action} interest rates by {bps} basis points today, citing {reason}. Markets reacted with {reaction}.",
            "symbols": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "JPM", "BAC", "GS"],
            "sentiment_range": (-0.6, 0.6)
        },
        {
            "title": "{company} Announces {type} of {target} for ${amount}B",
            "summary": "{company} has announced plans to {type} {target} for ${amount} billion in a move that {impact} its market position in the {industry} sector.",
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "GS", "PFE"],
            "sentiment_range": (-0.5, 0.9)
        },
        {
            "title": "Regulatory Concerns Mount for {sector} Stocks",
            "summary": "Regulatory scrutiny is increasing for companies in the {sector} sector, with {agency} considering new rules that could {impact} profitability and growth prospects.",
            "symbols": ["META", "GOOGL", "AMZN", "JPM", "BAC", "GS", "PFE", "JNJ", "UNH", "XOM"],
            "sentiment_range": (-0.9, -0.1)
        },
        {
            "title": "{country} Economic Data Shows {trend} in {metric}",
            "summary": "Latest economic data from {country} indicates a {trend} in {metric}, which could {impact} global markets and trade relationships.",
            "symbols": ["SPY", "EWJ", "FXI", "EWG", "EWU", "EWQ", "EWL", "EWA", "EWC", "RSX"],
            "sentiment_range": (-0.7, 0.7)
        }
    ]
    
    # Select a random news template
    template = random.choice(news_templates)
    
    # Fill in the template with random values
    if "company" in template["title"]:
        companies = ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "JPMorgan", "Bank of America", "Goldman Sachs", "Morgan Stanley", "Pfizer", "Johnson & Johnson", "UnitedHealth", "Exxon Mobil"]
        company = random.choice(companies)
    
    if "{adj}" in template["title"]:
        adj = random.choice(["Better-Than-Expected", "Worse-Than-Expected", "Mixed", "Record", "Disappointing", "Strong", "Weak"])
        comp = "beating" if adj in ["Better-Than-Expected", "Record", "Strong"] else "missing" if adj in ["Worse-Than-Expected", "Disappointing", "Weak"] else "matching"
    
    if "{quarter}" in template["title"]:
        quarter = random.randint(1, 4)
    
    if "{eps}" in template["summary"]:
        eps = round(random.uniform(0.5, 5.0), 2)
        exp_eps = round(eps * random.uniform(0.8, 1.2), 2)
    
    if "{revenue}" in template["summary"]:
        revenue = round(random.uniform(5, 100), 1)
        exp_revenue = round(revenue * random.uniform(0.9, 1.1), 1)
    
    if "{action}" in template["title"]:
        action = random.choice(["Raises", "Cuts", "Maintains", "Signals Future Hikes in", "Hints at Potential Cuts to"])
    
    if "{bps}" in template["title"]:
        bps = random.choice([25, 50, 75, 100])
    
    if "{reason}" in template["summary"]:
        reason = random.choice(["persistent inflation concerns", "slowing economic growth", "strong labor market data", "financial stability risks", "global economic uncertainties"])
    
    if "{reaction}" in template["summary"]:
        reaction = random.choice(["a strong rally", "increased volatility", "a selloff in equities", "a flattening yield curve", "mixed trading as investors digest the news"])
    
    if "{type}" in template["title"]:
        deal_type = random.choice(["Acquisition", "Merger", "Takeover", "Strategic Investment in", "Majority Stake Purchase in"])
    
    if "{target}" in template["title"]:
        targets = ["a Major Competitor", "a Tech Startup", "a Healthcare Provider", "a Financial Services Firm", "a Manufacturing Company", "a Retail Chain", "a Media Company"]
        target = random.choice(targets)
    
    if "{amount}" in template["title"]:
        amount = round(random.uniform(1, 50), 1)
    
    if "{impact}" in template["summary"]:
        impact = random.choice(["significantly strengthens", "moderately improves", "slightly enhances", "could potentially weaken", "raises questions about"])
    
    if "{industry}" in template["summary"]:
        industry = random.choice(["technology", "healthcare", "financial", "energy", "consumer", "industrial", "telecommunications", "media", "retail"])
    
    if "{sector}" in template["title"]:
        sector = random.choice(["Technology", "Banking", "Healthcare", "Energy", "Social Media", "E-commerce", "Pharmaceutical", "Automotive", "Telecommunications"])
    
    if "{agency}" in template["summary"]:
        agency = random.choice(["the SEC", "the FTC", "the DOJ", "the Federal Reserve", "the CFPB", "the FDA", "the EPA", "Congress", "the European Commission"])
    
    if "{country}" in template["title"]:
        country = random.choice(["US", "China", "Japan", "Germany", "UK", "France", "Italy", "Canada", "Australia", "Russia", "India", "Brazil"])
    
    if "{trend}" in template["title"]:
        trend = random.choice(["Improvement", "Deterioration", "Unexpected Growth", "Surprising Contraction", "Stability", "Increasing Volatility", "Resilience"])
    
    if "{metric}" in template["title"]:
        metric = random.choice(["GDP Growth", "Inflation", "Unemployment", "Consumer Spending", "Manufacturing Output", "Housing Market", "Trade Balance", "Business Confidence"])
    
    # Format the title and summary with the random values
    title = template["title"]
    summary = template["summary"]
    
    for var in ["company", "adj", "quarter", "action", "bps", "type", "target", "amount", "sector", "country", "trend", "metric"]:
        if "{" + var + "}" in title and var in locals():
            title = title.replace("{" + var + "}", str(locals()[var]))
    
    for var in ["company", "adj", "quarter", "eps", "exp_eps", "revenue", "exp_revenue", "comp", "action", "bps", "reason", "reaction", "type", "target", "amount", "impact", "industry", "sector", "agency", "country", "trend", "metric"]:
        if "{" + var + "}" in summary and var in locals():
            summary = summary.replace("{" + var + "}", str(locals()[var]))
    
    # Select random symbols from the template's symbol list
    num_symbols = random.randint(1, min(5, len(template["symbols"])))
    symbols = random.sample(template["symbols"], num_symbols)
    
    # Generate sentiment score
    sentiment_range = template["sentiment_range"]
    sentiment = round(random.uniform(sentiment_range[0], sentiment_range[1]), 2)
    
    # Create news item
    news_item = NewsItem(
        id=f"news-{int(time.time())}-{random.randint(1000, 9999)}",
        title=title,
        summary=summary,
        url=f"https://example.com/news/{int(time.time())}-{random.randint(1000, 9999)}",
        source=random.choice(["Bloomberg", "Reuters", "CNBC", "Financial Times", "Wall Street Journal", "MarketWatch"]),
        symbols=symbols,
        sentiment=sentiment,
        timestamp=datetime.utcnow()
    )
    
    # Update cache (keep only the latest 100 news items)
    news_cache.append(news_item)
    if len(news_cache) > 100:
        news_cache.pop(0)
    
    return news_item

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # client_id -> list of symbols
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = []
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    def subscribe(self, client_id: str, symbol: str):
        if client_id in self.subscriptions:
            if symbol not in self.subscriptions[client_id]:
                self.subscriptions[client_id].append(symbol)
                logger.info(f"Client {client_id} subscribed to {symbol}")
    
    def unsubscribe(self, client_id: str, symbol: str):
        if client_id in self.subscriptions and symbol in self.subscriptions[client_id]:
            self.subscriptions[client_id].remove(symbol)
            logger.info(f"Client {client_id} unsubscribed from {symbol}")
    
    async def broadcast_market_data(self, symbol: str, data: MarketDataPoint):
        for client_id, symbols in self.subscriptions.items():
            if symbol in symbols and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(data.dict())
                except Exception as e:
                    logger.error(f"Error sending market data to client {client_id}: {e}")
                    self.disconnect(client_id)
    
    async def broadcast_order_book(self, symbol: str, data: OrderBook):
        for client_id, symbols in self.subscriptions.items():
            if symbol in symbols and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(data.dict())
                except Exception as e:
                    logger.error(f"Error sending order book to client {client_id}: {e}")
                    self.disconnect(client_id)
    
    async def broadcast_news(self, news_item: NewsItem):
        for client_id, symbols in self.subscriptions.items():
            # Send news to clients subscribed to any of the symbols in the news item
            if any(symbol in symbols for symbol in news_item.symbols) and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(news_item.dict())
                except Exception as e:
                    logger.error(f"Error sending news to client {client_id}: {e}")
                    self.disconnect(client_id)

# Initialize connection manager
manager = ConnectionManager()

# WebSocket endpoint for real-time data
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle subscription/unsubscription requests
            if message.get("action") == "subscribe":
                symbol = message.get("symbol")
                if symbol:
                    manager.subscribe(client_id, symbol)
                    # Send the latest data immediately after subscription
                    if symbol in market_data_cache:
                        await websocket.send_json(market_data_cache[symbol].dict())
                    if symbol in order_book_cache:
                        await websocket.send_json(order_book_cache[symbol].dict())
            
            elif message.get("action") == "unsubscribe":
                symbol = message.get("symbol")
                if symbol:
                    manager.unsubscribe(client_id, symbol)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

# Background tasks for generating and broadcasting data
async def market_data_generator():
    """Background task to generate and broadcast market data"""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "GS", "MS", 
               "PFE", "JNJ", "UNH", "XOM", "CVX", "PG", "KO", "PEP", "WMT", "HD"]
    
    while True:
        for symbol in symbols:
            try:
                # Generate market data
                data = await generate_mock_market_data(symbol)
                
                # Broadcast to subscribed clients
                await manager.broadcast_market_data(symbol, data)
                
                # Add some randomness to avoid all symbols updating at exactly the same time
                await asyncio.sleep(random.uniform(0.05, 0.2))
            except Exception as e:
                logger.error(f"Error generating market data for {symbol}: {e}")
        
        # Wait before the next round of updates
        await asyncio.sleep(1)

async def order_book_generator():
    """Background task to generate and broadcast order book updates"""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "GS", "MS"]
    
    while True:
        for symbol in symbols:
            try:
                # Generate order book
                data = await generate_mock_order_book(symbol)
                
                # Broadcast to subscribed clients
                await manager.broadcast_order_book(symbol, data)
                
                # Add some randomness to avoid all symbols updating at exactly the same time
                await asyncio.sleep(random.uniform(0.1, 0.3))
            except Exception as e:
                logger.error(f"Error generating order book for {symbol}: {e}")
        
        # Wait before the next round of updates
        await asyncio.sleep(2)

async def news_generator():
    """Background task to generate and broadcast news items"""
    while True:
        try:
            # Generate news at random intervals
            await asyncio.sleep(random.uniform(5, 15))
            
            # Generate news item
            news_item = await generate_mock_news()
            
            # Broadcast to subscribed clients
            await manager.broadcast_news(news_item)
            
        except Exception as e:
            logger.error(f"Error generating news: {e}")

# API endpoints for historical data
@app.get("/historical/{symbol}")
async def get_historical_data(symbol: str, period: str = "1d", interval: str = "1m"):
    """Get historical market data for a symbol"""
    try:
        # In a real system, this would fetch data from a database or external API
        # For this mock, we'll generate synthetic historical data
        
        # Define time periods in seconds
        periods = {
            "1d": 24 * 60 * 60,
            "5d": 5 * 24 * 60 * 60,
            "1m": 30 * 24 * 60 * 60,
            "3m": 90 * 24 * 60 * 60,
            "6m": 180 * 24 * 60 * 60,
            "1y": 365 * 24 * 60 * 60,
            "5y": 5 * 365 * 24 * 60 * 60
        }
        
        # Define intervals in seconds
        intervals = {
            "1m": 60,
            "5m": 5 * 60,
            "15m": 15 * 60,
            "30m": 30 * 60,
            "1h": 60 * 60,
            "4h": 4 * 60 * 60,
            "1d": 24 * 60 * 60
        }
        
        if period not in periods:
            return {"error": f"Invalid period: {period}. Valid periods: {list(periods.keys())}"}
        
        if interval not in intervals:
            return {"error": f"Invalid interval: {interval}. Valid intervals: {list(intervals.keys())}"}
        
        # Calculate number of data points
        period_seconds = periods.get(period, 24 * 60 * 60)
        interval_seconds = intervals.get(interval, 60)
        num_points = min(10000, period_seconds // interval_seconds)  # Cap at 10,000 points
        
        # Generate synthetic data
        now = time.time()
        
        # Use the sum of ASCII values to generate different but consistent prices per symbol
        base_value = sum(ord(c) for c in symbol) % 1000
        
        # Generate time series with realistic price movements
        timestamps = [now - i * interval_seconds for i in range(num_points)]
        timestamps.reverse()  # Oldest first
        
        # Generate price series with random walk and some mean reversion
        prices = [base_value]
        for i in range(1, num_points):
            # Random walk with drift and volatility based on symbol
            volatility = base_value * 0.0005  # 0.05% daily volatility
            drift = base_value * 0.00005  # Small upward drift (0.005% per period)
            
            # Add some mean reversion
            mean_reversion = 0.001 * (base_value - prices[-1])
            
            # Calculate price change
            price_change = drift + mean_reversion + random.normalvariate(0, volatility)
            
            # Add to price series
            new_price = max(0.01, prices[-1] + price_change)  # Ensure price doesn't go negative
            prices.append(new_price)
        
        # Generate volume with some randomness and autocorrelation
        base_volume = base_value * 10
        volumes = [max(10, base_volume * (1 + 0.5 * random.normalvariate(0, 0.2)))]
        for i in range(1, num_points):
            # Volume has autocorrelation and some randomness
            autocorrelation = 0.7
            new_volume = (
                autocorrelation * volumes[-1] + 
                (1 - autocorrelation) * base_volume * (1 + 0.5 * random.normalvariate(0, 0.3))
            )
            volumes.append(max(10, new_volume))  # Ensure volume is positive
        
        # Format data as list of dictionaries
        data = []
        for i in range(num_points):
            data.append({
                "timestamp": datetime.fromtimestamp(timestamps[i]).isoformat(),
                "open": round(prices[i], 4),
                "high": round(prices[i] * (1 + random.uniform(0, 0.005)), 4),
                "low": round(prices[i] * (1 - random.uniform(0, 0.005)), 4),
                "close": round(prices[i], 4),
                "volume": round(volumes[i], 2)
            })
        
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data": data
        }
    
    except Exception as e:
        logger.error(f"Error generating historical data for {symbol}: {e}")
        return {"error": str(e)}

@app.get("/news")
async def get_news(symbols: Optional[str] = None, limit: int = 20):
    """Get recent news items, optionally filtered by symbols"""
    try:
        filtered_news = news_cache
        
        # Filter by symbols if provided
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(",")]
            filtered_news = [
                news for news in news_cache
                if any(symbol in news.symbols for symbol in symbol_list)
            ]
        
        # Sort by timestamp (newest first) and limit
        sorted_news = sorted(filtered_news, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return {"news": [news.dict() for news in sorted_news]}
    
    except Exception as e:
        logger.error(f"Error retrieving news: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "market_data_cache_size": len(market_data_cache),
        "order_book_cache_size": len(order_book_cache),
        "news_cache_size": len(news_cache),
        "active_connections": len(manager.active_connections)
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    # Start background tasks
    asyncio.create_task(market_data_generator())
    asyncio.create_task(order_book_generator())
    asyncio.create_task(news_generator())
    logger.info("Data ingestion service started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Data ingestion service shutting down")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
