import asyncio
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trading_engine")

# Initialize FastAPI app
app = FastAPI(title="FinTech Trading Engine")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(str, Enum):
    GTC = "good_till_canceled"  # Good Till Canceled
    IOC = "immediate_or_cancel"  # Immediate Or Cancel
    FOK = "fill_or_kill"         # Fill Or Kill
    DAY = "day"                  # Day Order

class AssetType(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    BOND = "bond"

# Models
class OrderRequest(BaseModel):
    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    asset_type: AssetType
    
    @validator('price')
    def price_required_for_limit_orders(cls, v, values):
        if 'order_type' in values and values['order_type'] in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Price is required for limit orders')
        return v
    
    @validator('stop_price')
    def stop_price_required_for_stop_orders(cls, v, values):
        if 'order_type' in values and values['order_type'] in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and v is None:
            raise ValueError('Stop price is required for stop orders')
        return v

class Order(OrderRequest):
    id: str
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    filled_quantity: float = 0
    average_fill_price: Optional[float] = None
    commission: float = 0
    user_id: str

class OrderUpdate(BaseModel):
    id: str
    status: OrderStatus
    filled_quantity: float
    average_fill_price: Optional[float] = None
    timestamp: datetime

class Position(BaseModel):
    user_id: str
    symbol: str
    quantity: float
    average_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_percent: float
    realized_pl: float
    asset_type: AssetType
    updated_at: datetime

class Portfolio(BaseModel):
    user_id: str
    total_value: float
    cash_balance: float
    positions_value: float
    day_pl: float
    day_pl_percent: float
    total_pl: float
    total_pl_percent: float
    positions: List[Position]
    updated_at: datetime

class RiskMetrics(BaseModel):
    user_id: str
    portfolio_var: float  # Value at Risk (1-day, 95%)
    portfolio_beta: float
    portfolio_sharpe: float
    portfolio_sortino: float
    portfolio_max_drawdown: float
    position_concentration: Dict[str, float]  # symbol -> % of portfolio
    sector_concentration: Dict[str, float]    # sector -> % of portfolio
    updated_at: datetime

# In-memory data stores (in production, use a database)
orders_db = {}  # order_id -> Order
positions_db = {}  # user_id -> {symbol -> Position}
portfolios_db = {}  # user_id -> Portfolio
risk_metrics_db = {}  # user_id -> RiskMetrics
market_prices = {}  # symbol -> price

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        logger.info(f"User {user_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_order_update(self, user_id: str, order_update: OrderUpdate):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(order_update.dict())
            except Exception as e:
                logger.error(f"Error sending order update to user {user_id}: {e}")
                self.disconnect(user_id)
    
    async def send_portfolio_update(self, user_id: str, portfolio: Portfolio):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(portfolio.dict())
            except Exception as e:
                logger.error(f"Error sending portfolio update to user {user_id}: {e}")
                self.disconnect(user_id)

# Initialize connection manager
manager = ConnectionManager()

# Mock market data
def initialize_mock_market_data():
    """Initialize mock market data for testing"""
    symbols = [
        # Stocks
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "GS", "MS",
        # ETFs
        "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "BND", "AGG", "GLD",
        # Crypto
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
        # Forex
        "EUR-USD", "GBP-USD", "JPY-USD", "CAD-USD", "AUD-USD"
    ]
    
    for symbol in symbols:
        # Generate a base price based on the symbol
        base_value = sum(ord(c) for c in symbol) % 1000
        
        if "USD" in symbol:  # Forex or Crypto
            if "BTC" in symbol:
                base_value = 30000 + random.uniform(-1000, 1000)
            elif "ETH" in symbol:
                base_value = 2000 + random.uniform(-100, 100)
            elif "-USD" in symbol and len(symbol) <= 7:  # Forex
                base_value = random.uniform(0.5, 2.0)
            else:  # Other crypto
                base_value = random.uniform(1, 100)
        elif symbol in ["SPY", "QQQ", "IWM", "VTI", "VOO"]:  # Major ETFs
            base_value = random.uniform(300, 500)
        else:  # Stocks
            base_value = random.uniform(50, 500)
        
        market_prices[symbol] = base_value

# Helper functions
def generate_order_id():
    """Generate a unique order ID"""
    return f"order-{uuid.uuid4()}"

def get_current_price(symbol: str) -> float:
    """Get the current market price for a symbol"""
    if symbol in market_prices:
        # Add some random price movement
        current_price = market_prices[symbol]
        price_change = current_price * random.uniform(-0.005, 0.005)  # +/- 0.5%
        new_price = max(0.01, current_price + price_change)
        market_prices[symbol] = new_price
        return new_price
    else:
        # If symbol not found, add it with a random price
        base_value = sum(ord(c) for c in symbol) % 1000
        price = base_value + random.uniform(-base_value * 0.1, base_value * 0.1)
        market_prices[symbol] = max(0.01, price)
        return market_prices[symbol]

async def process_order(order: Order, background_tasks: BackgroundTasks):
    """Process an order (mock execution)"""
    # Simulate order processing delay
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    # Get current market price
    current_price = get_current_price(order.symbol)
    
    # Determine if the order should be filled based on order type and price
    should_fill = False
    fill_price = current_price
    
    if order.order_type == OrderType.MARKET:
        # Market orders always fill
        should_fill = True
    
    elif order.order_type == OrderType.LIMIT:
        # Limit orders fill if the price is favorable
        if order.side == OrderSide.BUY and current_price <= order.price:
            should_fill = True
            fill_price = min(current_price, order.price)
        elif order.side == OrderSide.SELL and current_price >= order.price:
            should_fill = True
            fill_price = max(current_price, order.price)
    
    elif order.order_type == OrderType.STOP:
        # Stop orders become market orders when the stop price is reached
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            should_fill = True
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            should_fill = True
    
    elif order.order_type == OrderType.STOP_LIMIT:
        # Stop-limit orders become limit orders when the stop price is reached
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            # Check if the limit price is favorable
            if current_price <= order.price:
                should_fill = True
                fill_price = min(current_price, order.price)
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            # Check if the limit price is favorable
            if current_price >= order.price:
                should_fill = True
                fill_price = max(current_price, order.price)
    
    # Update order status
    if should_fill:
        # Calculate commission (mock)
        commission = order.quantity * fill_price * 0.001  # 0.1% commission
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = commission
        order.updated_at = datetime.utcnow()
        
        # Update position
        update_position(order)
        
        # Create order update for WebSocket
        order_update = OrderUpdate(
            id=order.id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_fill_price=fill_price,
            timestamp=datetime.utcnow()
        )
        
        # Send update via WebSocket
        background_tasks.add_task(manager.send_order_update, order.user_id, order_update)
        
        # Update portfolio and send update
        portfolio = calculate_portfolio(order.user_id)
        background_tasks.add_task(manager.send_portfolio_update, order.user_id, portfolio)
        
        logger.info(f"Order {order.id} filled at {fill_price}")
    else:
        # Order remains open
        order.status = OrderStatus.OPEN
        order.updated_at = datetime.utcnow()
        
        # Schedule order processing again after a delay
        background_tasks.add_task(schedule_order_processing, order)
        
        logger.info(f"Order {order.id} remains open")
    
    # Update the order in the database
    orders_db[order.id] = order
    
    return order

async def schedule_order_processing(order: Order):
    """Schedule order processing after a delay"""
    # Only process orders that are still open
    if order.status == OrderStatus.OPEN:
        # Wait for a random time
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        # Process the order again
        background_tasks = BackgroundTasks()
        await process_order(order, background_tasks)

def update_position(order: Order):
    """Update position after an order is filled"""
    user_id = order.user_id
    symbol = order.symbol
    
    # Initialize user positions if not exists
    if user_id not in positions_db:
        positions_db[user_id] = {}
    
    # Get current position or create new one
    if symbol in positions_db[user_id]:
        position = positions_db[user_id][symbol]
    else:
        position = Position(
            user_id=user_id,
            symbol=symbol,
            quantity=0,
            average_entry_price=0,
            current_price=get_current_price(symbol),
            market_value=0,
            unrealized_pl=0,
            unrealized_pl_percent=0,
            realized_pl=0,
            asset_type=order.asset_type,
            updated_at=datetime.utcnow()
        )
    
    # Calculate realized P&L for sells
    realized_pl = 0
    if order.side == OrderSide.SELL and position.quantity > 0:
        # Calculate realized P&L for the sold quantity
        realized_pl = (order.average_fill_price - position.average_entry_price) * min(order.quantity, position.quantity)
        position.realized_pl += realized_pl
    
    # Update position quantity and average entry price
    if order.side == OrderSide.BUY:
        # Calculate new average entry price
        new_quantity = position.quantity + order.quantity
        new_cost = (position.quantity * position.average_entry_price) + (order.quantity * order.average_fill_price)
        position.average_entry_price = new_cost / new_quantity if new_quantity > 0 else 0
        position.quantity = new_quantity
    else:  # SELL
        position.quantity -= order.quantity
    
    # Update current price and market value
    position.current_price = get_current_price(symbol)
    position.market_value = position.quantity * position.current_price
    
    # Calculate unrealized P&L
    if position.quantity > 0:
        position.unrealized_pl = (position.current_price - position.average_entry_price) * position.quantity
        position.unrealized_pl_percent = (position.current_price / position.average_entry_price - 1) * 100 if position.average_entry_price > 0 else 0
    else:
        position.unrealized_pl = 0
        position.unrealized_pl_percent = 0
    
    position.updated_at = datetime.utcnow()
    
    # Update or remove position
    if position.quantity > 0:
        positions_db[user_id][symbol] = position
    else:
        # Remove position if quantity is 0
        if symbol in positions_db[user_id]:
            del positions_db[user_id][symbol]

def calculate_portfolio(user_id: str) -> Portfolio:
    """Calculate portfolio metrics for a user"""
    # Get user positions
    user_positions = positions_db.get(user_id, {}).values()
    
    # Calculate portfolio metrics
    positions_value = sum(position.market_value for position in user_positions)
    
    # Get or initialize portfolio
    if user_id in portfolios_db:
        portfolio = portfolios_db[user_id]
    else:
        portfolio = Portfolio(
            user_id=user_id,
            total_value=100000,  # Initial account value
            cash_balance=100000,  # Initial cash balance
            positions_value=0,
            day_pl=0,
            day_pl_percent=0,
            total_pl=0,
            total_pl_percent=0,
            positions=[],
            updated_at=datetime.utcnow()
        )
    
    # Update portfolio
    portfolio.positions_value = positions_value
    portfolio.total_value = portfolio.cash_balance + positions_value
    
    # Calculate P&L (mock)
    portfolio.day_pl = sum(position.unrealized_pl for position in user_positions)
    portfolio.day_pl_percent = (portfolio.day_pl / portfolio.total_value) * 100 if portfolio.total_value > 0 else 0
    portfolio.total_pl = sum(position.unrealized_pl + position.realized_pl for position in user_positions)
    portfolio.total_pl_percent = (portfolio.total_pl / portfolio.total_value) * 100 if portfolio.total_value > 0 else 0
    
    # Update positions list
    portfolio.positions = list(user_positions)
    portfolio.updated_at = datetime.utcnow()
    
    # Update portfolio in database
    portfolios_db[user_id] = portfolio
    
    return portfolio

def calculate_risk_metrics(user_id: str) -> RiskMetrics:
    """Calculate risk metrics for a user's portfolio"""
    # Get user portfolio
    portfolio = portfolios_db.get(user_id)
    if not portfolio:
        return None
    
    # Get user positions
    user_positions = positions_db.get(user_id, {}).values()
    
    # Calculate position concentration
    total_value = portfolio.total_value
    position_concentration = {}
    for position in user_positions:
        position_concentration[position.symbol] = (position.market_value / total_value) * 100 if total_value > 0 else 0
    
    # Mock sector mapping
    sector_mapping = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "AMZN": "Consumer Cyclical",
        "META": "Technology",
        "TSLA": "Consumer Cyclical",
        "JPM": "Financial Services",
        "BAC": "Financial Services",
        "GS": "Financial Services",
        "MS": "Financial Services",
        "SPY": "ETF",
        "QQQ": "ETF",
        "IWM": "ETF",
        "VTI": "ETF",
        "VOO": "ETF",
        "VEA": "ETF",
        "VWO": "ETF",
        "BND": "ETF",
        "AGG": "ETF",
        "GLD": "ETF",
        "BTC-USD": "Cryptocurrency",
        "ETH-USD": "Cryptocurrency",
        "SOL-USD": "Cryptocurrency",
        "ADA-USD": "Cryptocurrency",
        "DOT-USD": "Cryptocurrency",
        "EUR-USD": "Forex",
        "GBP-USD": "Forex",
        "JPY-USD": "Forex",
        "CAD-USD": "Forex",
        "AUD-USD": "Forex"
    }
    
    # Calculate sector concentration
    sector_concentration = {}
    for position in user_positions:
        sector = sector_mapping.get(position.symbol, "Other")
        if sector not in sector_concentration:
            sector_concentration[sector] = 0
        sector_concentration[sector] += (position.market_value / total_value) * 100 if total_value > 0 else 0
    
    # Calculate mock risk metrics
    # In a real system, these would be calculated using historical data and proper risk models
    
    # Value at Risk (VaR) - simple calculation
    portfolio_var = total_value * 0.02  # Assume 2% daily VaR at 95% confidence
    
    # Beta - weighted average of position betas
    position_betas = {
        "AAPL": 1.2, "MSFT": 1.1, "GOOGL": 1.3, "AMZN": 1.4, "META": 1.5,
        "TSLA": 1.8, "JPM": 1.1, "BAC": 1.2, "GS": 1.3, "MS": 1.2,
        "SPY": 1.0, "QQQ": 1.2, "IWM": 1.3, "VTI": 1.0, "VOO": 1.0,
        "VEA": 0.9, "VWO": 1.1, "BND": 0.2, "AGG": 0.2, "GLD": 0.0,
        "BTC-USD": 2.5, "ETH-USD": 3.0, "SOL-USD": 3.5, "ADA-USD": 3.2, "DOT-USD": 3.3,
        "EUR-USD": 0.3, "GBP-USD": 0.4, "JPY-USD": 0.2, "CAD-USD": 0.5, "AUD-USD": 0.6
    }
    
    weighted_beta = 0
    for position in user_positions:
        beta = position_betas.get(position.symbol, 1.0)
        weight = position.market_value / total_value if total_value > 0 else 0
        weighted_beta += beta * weight
    
    # Sharpe and Sortino ratios - mock values
    portfolio_sharpe = random.uniform(0.5, 2.5)
    portfolio_sortino = random.uniform(0.7, 3.0)
    
    # Max drawdown - mock value
    portfolio_max_drawdown = random.uniform(5, 25)
    
    # Create risk metrics
    risk_metrics = RiskMetrics(
        user_id=user_id,
        portfolio_var=portfolio_var,
        portfolio_beta=weighted_beta,
        portfolio_sharpe=portfolio_sharpe,
        portfolio_sortino=portfolio_sortino,
        portfolio_max_drawdown=portfolio_max_drawdown,
        position_concentration=position_concentration,
        sector_concentration=sector_concentration,
        updated_at=datetime.utcnow()
    )
    
    # Update risk metrics in database
    risk_metrics_db[user_id] = risk_metrics
    
    return risk_metrics

# Routes
@app.post("/orders", response_model=Order)
async def create_order(order_request: OrderRequest, background_tasks: BackgroundTasks):
    """Create a new order"""
    try:
        # Generate order ID
        order_id = generate_order_id()
        
        # Create order
        order = Order(
            id=order_id,
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            user_id="demo_user",  # In a real system, this would come from auth
            **order_request.dict()
        )
        
        # Store order
        orders_db[order_id] = order
        
        # Process order asynchronously
        background_tasks.add_task(process_order, order, background_tasks)
        
        return order
    
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{order_id}", response_model=Order)
async def get_order(order_id: str):
    """Get order by ID"""
    if order_id in orders_db:
        return orders_db[order_id]
    raise HTTPException(status_code=404, detail="Order not found")

@app.get("/orders", response_model=List[Order])
async def list_orders(
    user_id: str = "demo_user",
    status: Optional[OrderStatus] = None,
    symbol: Optional[str] = None,
    limit: int = 50
):
    """List orders with optional filters"""
    # Filter orders
    filtered_orders = []
    for order in orders_db.values():
        if order.user_id != user_id:
            continue
        
        if status and order.status != status:
            continue
        
        if symbol and order.symbol != symbol:
            continue
        
        filtered_orders.append(order)
    
    # Sort by created_at (newest first)
    filtered_orders.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply limit
    return filtered_orders[:limit]

@app.delete("/orders/{order_id}", response_model=Order)
async def cancel_order(order_id: str):
    """Cancel an open order"""
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")
    
    order = orders_db[order_id]
    
    # Can only cancel open or pending orders
    if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel order with status {order.status}")
    
    # Update order status
    order.status = OrderStatus.CANCELED
    order.updated_at = datetime.utcnow()
    
    return order

@app.get("/portfolio", response_model=Portfolio)
async def get_portfolio(user_id: str = "demo_user"):
    """Get user portfolio"""
    # Calculate portfolio
    portfolio = calculate_portfolio(user_id)
    
    return portfolio

@app.get("/positions", response_model=List[Position])
async def list_positions(user_id: str = "demo_user"):
    """List user positions"""
    # Get user positions
    user_positions = positions_db.get(user_id, {}).values()
    
    return list(user_positions)

@app.get("/risk", response_model=RiskMetrics)
async def get_risk_metrics(user_id: str = "demo_user"):
    """Get risk metrics for a user's portfolio"""
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(user_id)
    
    if not risk_metrics:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return risk_metrics

@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    price = get_current_price(symbol)
    
    return {
        "symbol": symbol,
        "price": price,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, user_id)
    try:
        # Send initial portfolio data
        portfolio = calculate_portfolio(user_id)
        await websocket.send_json(portfolio.dict())
        
        # Keep connection alive
        while True:
            # Wait for messages (not used in this example, but required to keep connection open)
            data = await websocket.receive_text()
            
            # In a real system, you might handle commands from the client here
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "orders_count": len(orders_db),
        "users_count": len(portfolios_db)
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    # Initialize mock market data
    initialize_mock_market_data()
    logger.info("Trading engine started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Trading engine shutting down")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
