#!/usr/bin/env python3
"""
WhatsApp Webhook Server for UbuntuAI

Standalone Flask server to handle WhatsApp messages via Twilio.
This can be deployed separately from the main Streamlit app.

Usage:
    python whatsapp_webhook.py

Environment Variables Required:
    - OPENAI_API_KEY
    - TWILIO_ACCOUNT_SID
    - TWILIO_AUTH_TOKEN
    - TWILIO_WHATSAPP_NUMBER
"""

import os
import sys
from flask import Flask, request, jsonify
from datetime import datetime
import logging
from threading import Thread
import schedule
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our WhatsApp agent
from api.whatsapp_agent import WhatsAppBusinessAgent
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Initialize WhatsApp agent
try:
    whatsapp_agent = WhatsAppBusinessAgent()
    logger.info("WhatsApp agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize WhatsApp agent: {e}")
    whatsapp_agent = None

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "service": "UbuntuAI WhatsApp Bot",
        "status": "healthy" if whatsapp_agent else "error",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/webhook', methods=['GET', 'POST'])
def whatsapp_webhook():
    """
    Main webhook endpoint for WhatsApp messages
    
    GET: Webhook verification (required by Twilio)
    POST: Handle incoming messages
    """
    
    if request.method == 'GET':
        # Webhook verification
        return "WhatsApp webhook verified", 200
    
    elif request.method == 'POST':
        if not whatsapp_agent:
            logger.error("WhatsApp agent not initialized")
            return jsonify({"error": "Service unavailable"}), 503
        
        try:
            # Extract message data
            from_number = request.values.get('From', '').replace('whatsapp:', '')
            message_body = request.values.get('Body', '').strip()
            message_sid = request.values.get('MessageSid', '')
            
            # Log incoming message (without content for privacy)
            logger.info(f"Received message from {from_number[:8]}... (SID: {message_sid})")
            
            if not from_number or not message_body:
                logger.warning("Missing required fields in webhook request")
                return jsonify({"error": "Missing required fields"}), 400
            
            # Process message through agent
            response_text = whatsapp_agent.handle_message(from_number, message_body)
            
            # Send response back via Twilio
            to_number = f"whatsapp:{from_number}"
            
            message = whatsapp_agent.twilio_client.messages.create(
                body=response_text,
                from_=f"whatsapp:{settings.TWILIO_WHATSAPP_NUMBER}",
                to=to_number
            )
            
            logger.info(f"Response sent to {from_number[:8]}... (SID: {message.sid})")
            
            return jsonify({
                "status": "success",
                "message_sid": message.sid,
                "response_length": len(response_text)
            }), 200
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp message: {str(e)}")
            
            # Try to send error message to user
            try:
                error_response = whatsapp_agent._get_error_response()
                whatsapp_agent.twilio_client.messages.create(
                    body=error_response,
                    from_=f"whatsapp:{settings.TWILIO_WHATSAPP_NUMBER}",
                    to=f"whatsapp:{from_number}"
                )
            except:
                pass  # Don't fail if error message fails to send
            
            return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    if not whatsapp_agent:
        return jsonify({"error": "Service unavailable"}), 503
    
    stats = {
        "active_sessions": len(whatsapp_agent.user_sessions),
        "total_messages_processed": sum(
            session.get('message_count', 0) 
            for session in whatsapp_agent.user_sessions.values()
        ),
        "uptime": "Available", # In production, track actual uptime
        "last_cleanup": "Not implemented",
        "service_status": "healthy"
    }
    
    return jsonify(stats)

@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually trigger session cleanup"""
    if not whatsapp_agent:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        old_count = len(whatsapp_agent.user_sessions)
        whatsapp_agent.cleanup_old_sessions()
        new_count = len(whatsapp_agent.user_sessions)
        
        return jsonify({
            "status": "success",
            "sessions_cleaned": old_count - new_count,
            "remaining_sessions": new_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['POST'])
def test_message():
    """Test endpoint for debugging (development only)"""
    if not whatsapp_agent:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        test_number = request.json.get('from', 'test_user')
        test_message = request.json.get('message', 'Hello')
        
        response = whatsapp_agent.handle_message(test_number, test_message)
        
        return jsonify({
            "input": test_message,
            "output": response,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Background tasks
def cleanup_sessions():
    """Background task to cleanup old sessions"""
    if whatsapp_agent:
        try:
            whatsapp_agent.cleanup_old_sessions()
            logger.info("Session cleanup completed")
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")

def run_background_tasks():
    """Run scheduled background tasks"""
    schedule.every(1).hours.do(cleanup_sessions)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.before_first_request
def startup():
    """Initialize background tasks"""
    logger.info("Starting background task scheduler...")
    
    # Start background tasks in a separate thread
    background_thread = Thread(target=run_background_tasks, daemon=True)
    background_thread.start()

if __name__ == '__main__':
    # Validate environment
    required_env_vars = [
        'OPENAI_API_KEY',
        'TWILIO_ACCOUNT_SID', 
        'TWILIO_AUTH_TOKEN',
        'TWILIO_WHATSAPP_NUMBER'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    if not whatsapp_agent:
        logger.error("Failed to initialize WhatsApp agent")
        sys.exit(1)
    
    # Start server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting WhatsApp webhook server on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )