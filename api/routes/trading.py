from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import time
import uuid
import logging

# Import authentication utilities
from api.routes.auth import get_current_active_user, User

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

# Models
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    asset_class: AssetClass
    client_order_id: Optional[str] = None

class Order(OrderRequest):
    id: str
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    filled_quantity: float = 0
    average_fill_price: Optional[float] = None
    commission: Optional[float] = None
    user_id: str

class OrderResponse(BaseModel):
    order: Order
    message: str
    latency_ms: float

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

class Portfolio(BaseModel):
    positions: List[Position]
    total_value: float
    cash_balance: float
    buying_power: float
    margin_used: float
    margin_available: float

# Mock database of orders and positions (in production, use a real database)
orders_db = {}
positions_db = {}

# Utility functions
def generate_order_id():
    return str(uuid.uuid4())

def execute_order(order: Order):
    """
    Simulate order execution with ultra-low latency
    In a real system, this would connect to exchange APIs
    """
    # Simulate processing time (sub-millisecond)
    time.sleep(0.0005)  # 0.5 ms
    
    # For demo purposes, we'll just mark the order as filled
    order.status = OrderStatus.FILLED
    order.filled_quantity = order.quantity
    
    # Simulate a fill price (in a real system, this would come from the exchange)
    if order.order_type == OrderType.MARKET:
        # For market orders, simulate some slippage
        slippage = 0.001  # 0.1%
        direction = 1 if order.side == OrderSide.BUY else -1
        # In a real system, we would get the current market price
        market_price = 100.0  # Placeholder
        order.average_fill_price = market_price * (1 + direction * slippage)
    else:
        # For limit orders, use the specified price
        order.average_fill_price = order.price
    
    # Update the order in the database
    orders_db[order.id] = order
    
    # Update positions
    update_position(order)
    
    return order

def update_position(order: Order):
    """
    Update the user's position based on the executed order
    """
    user_id = order.user_id
    symbol = order.symbol
    
    # Get the user's positions
    if user_id not in positions_db:
        positions_db[user_id] = {}
    
    # Get the position for this symbol
    if symbol not in positions_db[user_id]:
        positions_db[user_id][symbol] = Position(
            symbol=symbol,
            quantity=0,
            average_entry_price=0,
            current_price=order.average_fill_price,
            unrealized_pnl=0,
            realized_pnl=0,
            asset_class=order.asset_class,
            market_value=0,
            cost_basis=0
        )
    
    position = positions_db[user_id][symbol]
    
    # Update the position based on the order
    if order.side == OrderSide.BUY:
        # Calculate the new average entry price
        total_cost = position.average_entry_price * position.quantity
        new_cost = order.average_fill_price * order.filled_quantity
        new_quantity = position.quantity + order.filled_quantity
        
        if new_quantity > 0:
            position.average_entry_price = (total_cost + new_cost) / new_quantity
        
        position.quantity += order.filled_quantity
    else:  # SELL
        # Calculate realized P&L
        if position.quantity > 0:
            realized_pnl = (order.average_fill_price - position.average_entry_price) * min(position.quantity, order.filled_quantity)
            position.realized_pnl += realized_pnl
        
        position.quantity -= order.filled_quantity
        
        # If position is now negative, reset the average entry price
        if position.quantity < 0:
            position.average_entry_price = order.average_fill_price
    
    # Update market value and unrealized P&L
    position.current_price = order.average_fill_price
    position.market_value = position.quantity * position.current_price
    position.cost_basis = position.quantity * position.average_entry_price
    position.unrealized_pnl = position.market_value - position.cost_basis
    
    # Update the position in the database
    positions_db[user_id][symbol] = position

@router.post("/orders", response_model=OrderResponse)
async def create_order(
    order_request: OrderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate a new order ID if not provided
    client_order_id = order_request.client_order_id or generate_order_id()
    
    # Create a new order
    order = Order(
        **order_request.dict(),
        id=generate_order_id(),
        status=OrderStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        user_id=current_user.username,
        client_order_id=client_order_id
    )
    
    # Store the order in the database
    orders_db[order.id] = order
    
    # Execute the order in the background for non-blocking API response
    background_tasks.add_task(execute_order, order)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the order
    logger.info(f"Order created: {order.id}, Symbol: {order.symbol}, Side: {order.side}, Type: {order.order_type}, Latency: {latency_ms:.3f}ms")
    
    # Check if latency exceeds threshold
    if latency_ms > 10:
        logger.warning(f"High latency detected: {latency_ms:.3f}ms for order {order.id}")
    
    return OrderResponse(
        order=order,
        message="Order received and being processed",
        latency_ms=latency_ms
    )

@router.get("/orders/{order_id}", response_model=Order)
async def get_order(
    order_id: str,
    current_user: User = Depends(get_current_active_user)
):
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")
    
    order = orders_db[order_id]
    
    # Check if the order belongs to the current user
    if order.user_id != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this order")
    
    return order

@router.get("/orders", response_model=List[Order])
async def list_orders(
    status: Optional[OrderStatus] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user)
):
    # Filter orders by user
    user_orders = [order for order in orders_db.values() if order.user_id == current_user.username]
    
    # Apply additional filters
    if status:
        user_orders = [order for order in user_orders if order.status == status]
    
    if symbol:
        user_orders = [order for order in user_orders if order.symbol == symbol]
    
    # Sort by creation time (newest first) and limit the results
    user_orders.sort(key=lambda x: x.created_at, reverse=True)
    user_orders = user_orders[:limit]
    
    return user_orders

@router.delete("/orders/{order_id}", response_model=OrderResponse)
async def cancel_order(
    order_id: str,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")
    
    order = orders_db[order_id]
    
    # Check if the order belongs to the current user
    if order.user_id != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to cancel this order")
    
    # Check if the order can be cancelled
    if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel order with status {order.status}")
    
    # Cancel the order
    order.status = OrderStatus.CANCELLED
    order.updated_at = datetime.utcnow()
    
    # Update the order in the database
    orders_db[order.id] = order
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the cancellation
    logger.info(f"Order cancelled: {order.id}, Latency: {latency_ms:.3f}ms")
    
    return OrderResponse(
        order=order,
        message="Order cancelled successfully",
        latency_ms=latency_ms
    )

@router.get("/portfolio", response_model=Portfolio)
async def get_portfolio(
    current_user: User = Depends(get_current_active_user)
):
    # Get the user's positions
    if current_user.username not in positions_db:
        positions_db[current_user.username] = {}
    
    positions = list(positions_db[current_user.username].values())
    
    # Calculate portfolio totals
    total_value = sum(position.market_value for position in positions)
    cash_balance = 100000.0  # Placeholder, in a real system this would come from a database
    
    # For demo purposes, we'll use simple margin calculations
    margin_used = total_value * 0.5  # Assuming 50% margin requirement
    margin_available = cash_balance - margin_used
    buying_power = cash_balance + margin_available
    
    return Portfolio(
        positions=positions,
        total_value=total_value,
        cash_balance=cash_balance,
        buying_power=buying_power,
        margin_used=margin_used,
        margin_available=margin_available
    )

@router.get("/positions", response_model=List[Position])
async def get_positions(
    asset_class: Optional[AssetClass] = None,
    current_user: User = Depends(get_current_active_user)
):
    # Get the user's positions
    if current_user.username not in positions_db:
        positions_db[current_user.username] = {}
    
    positions = list(positions_db[current_user.username].values())
    
    # Filter by asset class if specified
    if asset_class:
        positions = [position for position in positions if position.asset_class == asset_class]
    
    return positions
