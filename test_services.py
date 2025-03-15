#!/usr/bin/env python3
"""
Test script for AI-Powered FinTech Platform services.
This script tests the functionality of the LLM service, trading engine, and data ingestion service.
"""

import asyncio
import json
import logging
import requests
import time
import websockets
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_services")

# Service URLs
LLM_SERVICE_URL = "http://localhost:8002"
TRADING_ENGINE_URL = "http://localhost:8003"
DATA_INGESTION_URL = "http://localhost:8001"

async def test_llm_service():
    """Test the LLM service endpoints"""
    logger.info("Testing LLM Service...")
    
    # Test health check
    try:
        response = requests.get(f"{LLM_SERVICE_URL}/health")
        response.raise_for_status()
        logger.info(f"LLM Service health check: {response.json()}")
    except Exception as e:
        logger.error(f"LLM Service health check failed: {e}")
        return False
    
    # Test sentiment analysis
    try:
        payload = {
            "text": "The market is showing strong growth potential despite recent volatility. Earnings reports have exceeded expectations, and the Fed's recent policy decisions suggest a favorable environment for equities."
        }
        response = requests.post(f"{LLM_SERVICE_URL}/sentiment", json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Sentiment Analysis Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Sentiment analysis test failed: {e}")
        return False
    
    # Test document summarization
    try:
        payload = {
            "document": """
            Quarterly Financial Report - Q3 2023
            
            Executive Summary:
            The company has shown remarkable resilience in Q3 2023, with revenue growth of 15% year-over-year, exceeding analyst expectations of 12%. EBITDA margins improved by 200 basis points to 28%, driven by operational efficiencies and strategic cost management initiatives.
            
            Financial Highlights:
            - Revenue: $1.25 billion (up 15% YoY)
            - Gross Margin: 62% (up from 60% in Q3 2022)
            - EBITDA: $350 million (up 23% YoY)
            - Net Income: $220 million (up 18% YoY)
            - EPS: $2.45 (vs analyst consensus of $2.30)
            - Free Cash Flow: $280 million (22% of revenue)
            
            Segment Performance:
            1. Product A: Revenue up 20%, contributing 45% of total revenue
            2. Product B: Revenue up 10%, contributing 30% of total revenue
            3. Product C: Revenue up 12%, contributing 25% of total revenue
            
            Regional Performance:
            - North America: 50% of revenue, growth of 18% YoY
            - Europe: 30% of revenue, growth of 12% YoY
            - Asia-Pacific: 15% of revenue, growth of 22% YoY
            - Rest of World: 5% of revenue, growth of 8% YoY
            
            Outlook:
            Based on the strong performance in Q3 and current market conditions, we are raising our full-year guidance:
            - Revenue growth: 14-16% (previously 12-14%)
            - EBITDA margin: 27-29% (previously 26-28%)
            - EPS: $9.50-$9.80 (previously $9.20-$9.50)
            
            We remain cautiously optimistic about market conditions while monitoring macroeconomic factors including inflation, interest rates, and global supply chain dynamics.
            """,
            "max_length": 300,
            "focus_areas": ["financial performance", "outlook"]
        }
        response = requests.post(f"{LLM_SERVICE_URL}/summarize", json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Document Summary Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Document summarization test failed: {e}")
        return False
    
    # Test trading advice
    try:
        payload = {
            "portfolio": {
                "AAPL": 25.0,
                "MSFT": 20.0,
                "GOOGL": 15.0,
                "AMZN": 10.0,
                "BND": 30.0
            },
            "risk_tolerance": 0.7,
            "investment_horizon": "medium_term",
            "market_conditions": {
                "market_trend": "bullish",
                "volatility": "moderate",
                "interest_rates": "rising"
            }
        }
        response = requests.post(f"{LLM_SERVICE_URL}/trading-advice", json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Trading Advice Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Trading advice test failed: {e}")
        return False
    
    # Test fraud detection
    try:
        payload = {
            "transaction_data": {
                "transaction_id": "tx123456",
                "amount": 15000.00,
                "currency": "USD",
                "timestamp": datetime.utcnow().isoformat(),
                "merchant": "Unusual Electronics Store",
                "merchant_category": "Electronics",
                "payment_method": "credit_card",
                "card_present": False,
                "ip_address": "203.0.113.195",
                "device_id": "device_8765432"
            },
            "user_history": {
                "user_id": "user123",
                "account_age_days": 45,
                "typical_transaction_amount": 200.00,
                "typical_merchants": ["Amazon", "Walmart", "Target"],
                "typical_categories": ["Retail", "Groceries", "Dining"],
                "typical_locations": ["New York, USA"],
                "previous_flags": 0
            }
        }
        response = requests.post(f"{LLM_SERVICE_URL}/fraud-detection", json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Fraud Detection Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Fraud detection test failed: {e}")
        return False
    
    logger.info("LLM Service tests completed successfully")
    return True

async def test_trading_engine():
    """Test the trading engine endpoints"""
    logger.info("Testing Trading Engine...")
    
    # Test health check
    try:
        response = requests.get(f"{TRADING_ENGINE_URL}/health")
        response.raise_for_status()
        logger.info(f"Trading Engine health check: {response.json()}")
    except Exception as e:
        logger.error(f"Trading Engine health check failed: {e}")
        return False
    
    # Test creating an order
    try:
        payload = {
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market",
            "time_in_force": "good_till_canceled",
            "asset_type": "stock"
        }
        response = requests.post(f"{TRADING_ENGINE_URL}/orders", json=payload)
        response.raise_for_status()
        order = response.json()
        order_id = order["id"]
        logger.info(f"Created order: {json.dumps(order, indent=2)}")
        
        # Wait for order to be processed
        time.sleep(2)
        
        # Get order status
        response = requests.get(f"{TRADING_ENGINE_URL}/orders/{order_id}")
        response.raise_for_status()
        updated_order = response.json()
        logger.info(f"Updated order status: {updated_order['status']}")
    except Exception as e:
        logger.error(f"Order creation test failed: {e}")
        return False
    
    # Test getting portfolio
    try:
        response = requests.get(f"{TRADING_ENGINE_URL}/portfolio")
        response.raise_for_status()
        portfolio = response.json()
        logger.info(f"Portfolio: {json.dumps(portfolio, indent=2)}")
    except Exception as e:
        logger.error(f"Portfolio retrieval test failed: {e}")
        return False
    
    # Test getting positions
    try:
        response = requests.get(f"{TRADING_ENGINE_URL}/positions")
        response.raise_for_status()
        positions = response.json()
        logger.info(f"Positions: {json.dumps(positions, indent=2)}")
    except Exception as e:
        logger.error(f"Positions retrieval test failed: {e}")
        return False
    
    # Test getting risk metrics
    try:
        response = requests.get(f"{TRADING_ENGINE_URL}/risk")
        response.raise_for_status()
        risk_metrics = response.json()
        logger.info(f"Risk Metrics: {json.dumps(risk_metrics, indent=2)}")
    except Exception as e:
        logger.error(f"Risk metrics retrieval test failed: {e}")
        return False
    
    # Test getting market data
    try:
        response = requests.get(f"{TRADING_ENGINE_URL}/market-data/AAPL")
        response.raise_for_status()
        market_data = response.json()
        logger.info(f"Market Data for AAPL: {json.dumps(market_data, indent=2)}")
    except Exception as e:
        logger.error(f"Market data retrieval test failed: {e}")
        return False
    
    logger.info("Trading Engine tests completed successfully")
    return True

async def test_data_ingestion():
    """Test the data ingestion service endpoints"""
    logger.info("Testing Data Ingestion Service...")
    
    # Test health check
    try:
        response = requests.get(f"{DATA_INGESTION_URL}/health")
        response.raise_for_status()
        logger.info(f"Data Ingestion Service health check: {response.json()}")
    except Exception as e:
        logger.error(f"Data Ingestion Service health check failed: {e}")
        return False
    
    # Test getting historical data
    try:
        response = requests.get(f"{DATA_INGESTION_URL}/historical/AAPL?period=1d&interval=5m")
        response.raise_for_status()
        historical_data = response.json()
        logger.info(f"Historical Data for AAPL: {len(historical_data['data'])} data points")
        logger.info(f"Sample data point: {json.dumps(historical_data['data'][0], indent=2)}")
    except Exception as e:
        logger.error(f"Historical data retrieval test failed: {e}")
        return False
    
    # Test getting news
    try:
        response = requests.get(f"{DATA_INGESTION_URL}/news?symbols=AAPL,MSFT&limit=5")
        response.raise_for_status()
        news = response.json()
        news_items = news.get("news", [])
        logger.info(f"News: {len(news_items)} items")
        if news_items:
            logger.info(f"Sample news item: {json.dumps(news_items[0], indent=2)}")
        else:
            logger.error(f"News retrieval test failed: {len(news_items)}")
            return False
    except Exception as e:
        logger.error(f"News retrieval test failed: {e}")
        return False
    
    # Test WebSocket connection (brief test)
    try:
        logger.info("Testing WebSocket connection (will run for 5 seconds)...")
        
        async def websocket_test():
            uri = f"ws://localhost:8001/ws/test_client"
            async with websockets.connect(uri) as websocket:
                # Subscribe to a symbol
                await websocket.send(json.dumps({"action": "subscribe", "symbol": "AAPL"}))
                
                # Listen for updates for a short period
                start_time = time.time()
                while time.time() - start_time < 5:  # Run for 5 seconds
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        logger.info(f"Received WebSocket data: {json.dumps(data, indent=2)}")
                    except asyncio.TimeoutError:
                        continue
        
        await websocket_test()
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        return False
    
    logger.info("Data Ingestion Service tests completed successfully")
    return True

async def main():
    """Run all tests"""
    logger.info("Starting tests for AI-Powered FinTech Platform services")
    
    # Test all services
    llm_success = await test_llm_service()
    trading_success = await test_trading_engine()
    data_success = await test_data_ingestion()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"LLM Service: {'✅ PASSED' if llm_success else '❌ FAILED'}")
    logger.info(f"Trading Engine: {'✅ PASSED' if trading_success else '❌ FAILED'}")
    logger.info(f"Data Ingestion: {'✅ PASSED' if data_success else '❌ FAILED'}")
    
    if llm_success and trading_success and data_success:
        logger.info("All services are functioning correctly!")
    else:
        logger.warning("Some services failed the tests. Please check the logs for details.")

if __name__ == "__main__":
    asyncio.run(main())
