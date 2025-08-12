"""
Advanced Security Management System for UbuntuAI RAG
Implements industry-standard security practices, rate limiting, and threat detection
"""

import logging
import re
import hashlib
import hmac
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import threading
from collections import defaultdict, deque

# Security libraries
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event record"""
    timestamp: datetime
    event_type: str
    severity: str
    user_id: Optional[str]
    ip_address: Optional[str]
    details: Dict[str, Any]
    action_taken: str

@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    current_requests: int
    max_requests: int
    window_start: datetime
    window_duration: timedelta
    reset_time: datetime

class InputValidator:
    """Advanced input validation and sanitization"""
    
    def __init__(self):
        self.blocked_patterns = self._load_blocked_patterns()
        self.sanitization_rules = self._load_sanitization_rules()
        self.validation_rules = self._load_validation_rules()
    
    def _load_blocked_patterns(self) -> List[re.Pattern]:
        """Load blocked content patterns"""
        
        patterns = [
            # Script injection
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            
            # Code execution
            r"eval\s*\(",
            r"exec\s*\(",
            r"import\s+os",
            r"subprocess\.",
            r"__import__\s*\(",
            
            # SQL injection
            r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
            r"(\b(and|or)\s+\d+\s*=\s*\d+)",
            
            # Path traversal
            r"\.\./",
            r"\.\.\\",
            
            # Command injection
            r"(\||&|;|\$\(|\`)",
            
            # XSS patterns
            r"<iframe.*?>",
            r"<object.*?>",
            r"<embed.*?>",
            
            # Malicious URLs
            r"data:text/html",
            r"data:application/javascript",
            
            # Suspicious encodings
            r"\\x[0-9a-fA-F]{2}",
            r"\\u[0-9a-fA-F]{4}",
            r"\\[0-7]{3}",
            
            # Ghanaian business context violations
            r"(?!.*ghana|.*ghanian|.*accra|.*kumasi|.*fintech|.*agritech|.*healthtech).*",
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _load_sanitization_rules(self) -> Dict[str, Any]:
        """Load content sanitization rules"""
        
        return {
            "html_tags": {
                "allowed": ["b", "i", "em", "strong", "u", "br", "p"],
                "strip_attributes": True
            },
            "max_length": {
                "query": 2000,
                "response": 10000,
                "metadata": 1000
            },
            "encoding": "utf-8",
            "normalize_whitespace": True
        }
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules"""
        
        return {
            "query": {
                "min_length": 3,
                "max_length": 2000,
                "required_words": ["ghana", "startup", "business", "fintech", "agritech", "healthtech"],
                "max_words": 500
            },
            "user_context": {
                "allowed_sectors": ["fintech", "agritech", "healthtech"],
                "allowed_regions": settings.GHANA_REGIONS if hasattr(settings, 'GHANA_REGIONS') else [],
                "allowed_experience_levels": ["beginner", "intermediate", "advanced", "expert"]
            }
        }
    
    def validate_input(self, 
                      content: str, 
                      content_type: str = "query",
                      user_context: Dict[str, Any] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate and sanitize input content"""
        
        try:
            # Check for blocked patterns
            for pattern in self.blocked_patterns:
                if pattern.search(content):
                    return False, f"Content contains blocked pattern: {pattern.pattern}", {}
            
            # Length validation
            max_length = self.sanitization_rules["max_length"].get(content_type, 1000)
            if len(content) > max_length:
                return False, f"Content too long. Maximum length: {max_length}", {}
            
            if len(content) < self.validation_rules[content_type]["min_length"]:
                return False, f"Content too short. Minimum length: {self.validation_rules[content_type]['min_length']}", {}
            
            # Content-specific validation
            if content_type == "query":
                validation_result = self._validate_query(content, user_context)
                if not validation_result[0]:
                    return validation_result
            
            # Sanitize content
            sanitized_content = self._sanitize_content(content, content_type)
            
            # Additional security checks
            security_score = self._calculate_security_score(sanitized_content)
            
            return True, "Content validated successfully", {
                "sanitized_content": sanitized_content,
                "security_score": security_score,
                "validation_metadata": {
                    "original_length": len(content),
                    "sanitized_length": len(sanitized_content),
                    "content_type": content_type
                }
            }
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False, f"Validation error: {str(e)}", {}
    
    def _validate_query(self, query: str, user_context: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Validate query content"""
        
        # Check for required Ghanaian business context
        query_lower = query.lower()
        required_words = self.validation_rules["query"]["required_words"]
        
        # At least one required word should be present
        if not any(word in query_lower for word in required_words):
            return False, "Query must be related to Ghanaian startup ecosystem (fintech, agritech, healthtech)"
        
        # Word count validation
        word_count = len(query.split())
        if word_count > self.validation_rules["query"]["max_words"]:
            return False, f"Query too long. Maximum words: {self.validation_rules['query']['max_words']}"
        
        # Validate user context if provided
        if user_context:
            context_validation = self._validate_user_context(user_context)
            if not context_validation[0]:
                return context_validation
        
        return True, "Query validated successfully"
    
    def _validate_user_context(self, user_context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate user context"""
        
        # Validate sector
        if "sector" in user_context:
            sector = user_context["sector"]
            if sector not in self.validation_rules["user_context"]["allowed_sectors"]:
                return False, f"Invalid sector: {sector}. Allowed: {self.validation_rules['user_context']['allowed_sectors']}"
        
        # Validate region
        if "region" in user_context:
            region = user_context["region"]
            if region not in self.validation_rules["user_context"]["allowed_regions"]:
                return False, f"Invalid region: {region}. Allowed: {self.validation_rules['user_context']['allowed_regions']}"
        
        # Validate experience level
        if "experience_level" in user_context:
            experience = user_context["experience_level"]
            if experience not in self.validation_rules["user_context"]["allowed_experience_levels"]:
                return False, f"Invalid experience level: {experience}"
        
        return True, "User context validated successfully"
    
    def _sanitize_content(self, content: str, content_type: str) -> str:
        """Sanitize content based on type"""
        
        sanitized = content
        
        # HTML tag sanitization
        if self.sanitization_rules["html_tags"]["strip_attributes"]:
            # Remove HTML attributes
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Normalize whitespace
        if self.sanitization_rules["normalize_whitespace"]:
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Encoding normalization
        try:
            sanitized = sanitized.encode(self.sanitization_rules["encoding"]).decode(self.sanitization_rules["encoding"])
        except (UnicodeEncodeError, UnicodeDecodeError):
            sanitized = sanitized.encode('utf-8', errors='ignore').decode('utf-8')
        
        return sanitized
    
    def _calculate_security_score(self, content: str) -> float:
        """Calculate security score for content"""
        
        score = 1.0
        
        # Penalize suspicious patterns
        suspicious_patterns = [
            r'[<>]',  # HTML tags
            r'[&|;]',  # Command separators
            r'\\[xXu]',  # Hex/Unicode escapes
            r'[0-9]{16,}',  # Long numbers (potential credit cards)
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content):
                score -= 0.1
        
        # Reward Ghanaian business context
        ghana_keywords = ["ghana", "ghanian", "accra", "kumasi", "fintech", "agritech", "healthtech"]
        if any(keyword in content.lower() for keyword in ghana_keywords):
            score += 0.2
        
        return max(0.0, min(1.0, score))

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.limits = self._load_rate_limits()
        self.request_history = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips = {}
        self.rate_limit_lock = threading.Lock()
    
    def _load_rate_limits(self) -> Dict[str, Dict[str, Any]]:
        """Load rate limiting configuration"""
        
        return {
            "default": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
                "burst_limit": 10
            },
            "authenticated": {
                "requests_per_minute": 120,
                "requests_per_hour": 2000,
                "requests_per_day": 20000,
                "burst_limit": 20
            },
            "premium": {
                "requests_per_minute": 300,
                "requests_per_hour": 5000,
                "requests_per_day": 50000,
                "burst_limit": 50
            }
        }
    
    def check_rate_limit(self, 
                        identifier: str, 
                        limit_type: str = "default",
                        request_size: int = 1) -> Tuple[bool, RateLimitInfo]:
        """Check if request is within rate limits"""
        
        with self.rate_limit_lock:
            current_time = datetime.now()
            
            # Get rate limit configuration
            limits = self.limits.get(limit_type, self.limits["default"])
            
            # Check if IP is blocked
            if identifier in self.blocked_ips:
                block_info = self.blocked_ips[identifier]
                if current_time < block_info["until"]:
                    return False, RateLimitInfo(
                        current_requests=0,
                        max_requests=0,
                        window_start=current_time,
                        window_duration=timedelta(0),
                        reset_time=block_info["until"]
                    )
                else:
                    # Unblock IP
                    del self.blocked_ips[identifier]
            
            # Get request history
            history = self.request_history[identifier]
            
            # Remove old requests
            self._clean_old_requests(history, current_time)
            
            # Check various time windows
            minute_requests = self._count_requests_in_window(history, current_time, timedelta(minutes=1))
            hour_requests = self._count_requests_in_window(history, current_time, timedelta(hours=1))
            day_requests = self._count_requests_in_window(history, current_time, timedelta(days=1))
            
            # Check limits
            within_limits = (
                minute_requests + request_size <= limits["requests_per_minute"] and
                hour_requests + request_size <= limits["requests_per_hour"] and
                day_requests + request_size <= limits["requests_per_day"]
            )
            
            # Check burst limit
            if within_limits and minute_requests + request_size > limits["burst_limit"]:
                # Implement exponential backoff
                within_limits = False
            
            # Record request if within limits
            if within_limits:
                history.append({
                    "timestamp": current_time,
                    "size": request_size
                })
            
            # Calculate reset times
            minute_reset = current_time + timedelta(minutes=1)
            hour_reset = current_time + timedelta(hours=1)
            day_reset = current_time + timedelta(days=1)
            
            # Determine next reset time
            if minute_requests + request_size > limits["requests_per_minute"]:
                reset_time = minute_reset
            elif hour_requests + request_size > limits["requests_per_hour"]:
                reset_time = hour_reset
            elif day_requests + request_size > limits["requests_per_day"]:
                reset_time = day_reset
            else:
                reset_time = minute_reset
            
            # Block IP if limits exceeded significantly
            if not within_limits and minute_requests > limits["requests_per_minute"] * 2:
                self._block_ip(identifier, current_time)
            
            return within_limits, RateLimitInfo(
                current_requests=minute_requests,
                max_requests=limits["requests_per_minute"],
                window_start=current_time - timedelta(minutes=1),
                window_duration=timedelta(minutes=1),
                reset_time=reset_time
            )
    
    def _clean_old_requests(self, history: deque, current_time: datetime):
        """Remove old requests from history"""
        
        cutoff_time = current_time - timedelta(days=1)
        while history and history[0]["timestamp"] < cutoff_time:
            history.popleft()
    
    def _count_requests_in_window(self, 
                                 history: deque, 
                                 current_time: datetime, 
                                 window: timedelta) -> int:
        """Count requests in a specific time window"""
        
        cutoff_time = current_time - window
        count = 0
        
        for request in reversed(history):
            if request["timestamp"] >= cutoff_time:
                count += request["size"]
            else:
                break
        
        return count
    
    def _block_ip(self, identifier: str, current_time: datetime):
        """Block IP address temporarily"""
        
        # Exponential backoff: 1 minute, 2 minutes, 4 minutes, etc.
        current_block = self.blocked_ips.get(identifier, {"count": 0})
        block_count = current_block["count"] + 1
        block_duration = timedelta(minutes=2 ** (block_count - 1))
        
        self.blocked_ips[identifier] = {
            "until": current_time + block_duration,
            "count": block_count,
            "blocked_at": current_time
        }
        
        logger.warning(f"IP {identifier} blocked for {block_duration} due to rate limit violations")
    
    def get_rate_limit_info(self, identifier: str, limit_type: str = "default") -> RateLimitInfo:
        """Get current rate limit information for an identifier"""
        
        with self.rate_limit_lock:
            current_time = datetime.now()
            limits = self.limits.get(limit_type, self.limits["default"])
            history = self.request_history[identifier]
            
            self._clean_old_requests(history, current_time)
            
            minute_requests = self._count_requests_in_window(history, current_time, timedelta(minutes=1))
            
            return RateLimitInfo(
                current_requests=minute_requests,
                max_requests=limits["requests_per_minute"],
                window_start=current_time - timedelta(minutes=1),
                window_duration=timedelta(minutes=1),
                reset_time=current_time + timedelta(minutes=1)
            )

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.behavioral_analysis = self._load_behavioral_patterns()
        self.threat_history = defaultdict(list)
        self.risk_scores = defaultdict(float)
    
    def _load_threat_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Load threat detection patterns"""
        
        return {
            "high_risk": [
                re.compile(r"(\b(union|select|insert|update|delete|drop|create|alter)\b)", re.IGNORECASE),
                re.compile(r"(eval\s*\(|exec\s*\(|__import__\s*\()", re.IGNORECASE),
                re.compile(r"(<script.*?>.*?</script>)", re.IGNORECASE),
                re.compile(r"(javascript:|vbscript:)", re.IGNORECASE),
            ],
            "medium_risk": [
                re.compile(r"(\.\./|\.\.\\)", re.IGNORECASE),
                re.compile(r"(\||&|;|\$\(|\`)", re.IGNORECASE),
                re.compile(r"(data:text/html|data:application/javascript)", re.IGNORECASE),
            ],
            "low_risk": [
                re.compile(r"(\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4})", re.IGNORECASE),
                re.compile(r"([0-9]{16,})", re.IGNORECASE),
            ]
        }
    
    def _load_behavioral_patterns(self) -> Dict[str, Any]:
        """Load behavioral analysis patterns"""
        
        return {
            "rapid_requests": {
                "threshold": 10,
                "time_window": 60,  # seconds
                "risk_increase": 0.3
            },
            "repeated_queries": {
                "threshold": 5,
                "time_window": 300,  # seconds
                "risk_increase": 0.2
            },
            "large_requests": {
                "threshold": 1000,  # characters
                "risk_increase": 0.1
            }
        }
    
    def analyze_threat(self, 
                      content: str, 
                      user_id: Optional[str] = None,
                      ip_address: Optional[str] = None,
                      request_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze content for potential threats"""
        
        threat_analysis = {
            "risk_level": "low",
            "risk_score": 0.0,
            "threats_detected": [],
            "behavioral_risks": [],
            "recommendations": []
        }
        
        # Pattern-based threat detection
        for risk_level, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    threat_analysis["threats_detected"].append({
                        "type": "pattern_match",
                        "risk_level": risk_level,
                        "pattern": pattern.pattern,
                        "description": f"Content matches {risk_level} risk pattern"
                    })
                    
                    # Update risk score
                    if risk_level == "high_risk":
                        threat_analysis["risk_score"] += 0.5
                    elif risk_level == "medium_risk":
                        threat_analysis["risk_score"] += 0.3
                    else:
                        threat_analysis["risk_score"] += 0.1
        
        # Behavioral analysis
        if user_id or ip_address:
            identifier = user_id or ip_address
            behavioral_risks = self._analyze_behavior(identifier, request_metadata)
            threat_analysis["behavioral_risks"] = behavioral_risks
            
            # Update risk score based on behavior
            for risk in behavioral_risks:
                threat_analysis["risk_score"] += risk.get("risk_increase", 0.0)
        
        # Determine overall risk level
        if threat_analysis["risk_score"] >= 0.7:
            threat_analysis["risk_level"] = "high"
        elif threat_analysis["risk_score"] >= 0.4:
            threat_analysis["risk_level"] = "medium"
        else:
            threat_analysis["risk_level"] = "low"
        
        # Generate recommendations
        threat_analysis["recommendations"] = self._generate_recommendations(threat_analysis)
        
        # Update threat history
        if user_id or ip_address:
            identifier = user_id or ip_address
            self._update_threat_history(identifier, threat_analysis)
        
        return threat_analysis
    
    def _analyze_behavior(self, 
                         identifier: str, 
                         request_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze user behavior for suspicious patterns"""
        
        behavioral_risks = []
        current_time = datetime.now()
        
        # Get recent history
        recent_requests = self.threat_history[identifier]
        
        # Clean old history
        cutoff_time = current_time - timedelta(hours=1)
        recent_requests = [req for req in recent_requests if req["timestamp"] > cutoff_time]
        
        # Rapid requests analysis
        if len(recent_requests) > self.behavioral_patterns["rapid_requests"]["threshold"]:
            behavioral_risks.append({
                "type": "rapid_requests",
                "description": "User making requests too rapidly",
                "risk_increase": self.behavioral_patterns["rapid_requests"]["risk_increase"]
            })
        
        # Repeated queries analysis
        if request_metadata and "query" in request_metadata:
            query = request_metadata["query"]
            similar_queries = sum(1 for req in recent_requests 
                                if req.get("query") and self._similarity_score(query, req["query"]) > 0.8)
            
            if similar_queries > self.behavioral_patterns["repeated_queries"]["threshold"]:
                behavioral_risks.append({
                    "type": "repeated_queries",
                    "description": "User making similar repeated queries",
                    "risk_increase": self.behavioral_patterns["repeated_queries"]["risk_increase"]
                })
        
        # Large request analysis
        if request_metadata and "content_length" in request_metadata:
            content_length = request_metadata["content_length"]
            if content_length > self.behavioral_patterns["large_requests"]["threshold"]:
                behavioral_risks.append({
                    "type": "large_request",
                    "description": "Unusually large request content",
                    "risk_increase": self.behavioral_patterns["large_requests"]["risk_increase"]
                })
        
        return behavioral_risks
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate simple similarity score between two texts"""
        
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_recommendations(self, threat_analysis: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on threat analysis"""
        
        recommendations = []
        
        if threat_analysis["risk_level"] == "high":
            recommendations.extend([
                "Block user/IP address temporarily",
                "Review request logs for similar patterns",
                "Implement additional authentication",
                "Consider CAPTCHA or rate limiting"
            ])
        elif threat_analysis["risk_level"] == "medium":
            recommendations.extend([
                "Monitor user behavior closely",
                "Implement stricter rate limiting",
                "Add request validation",
                "Log all requests for analysis"
            ])
        else:
            recommendations.extend([
                "Continue normal monitoring",
                "Log request for audit trail"
            ])
        
        return recommendations
    
    def _update_threat_history(self, identifier: str, threat_analysis: Dict[str, Any]):
        """Update threat history for an identifier"""
        
        self.threat_history[identifier].append({
            "timestamp": datetime.now(),
            "risk_score": threat_analysis["risk_score"],
            "risk_level": threat_analysis["risk_level"],
            "threats_detected": len(threat_analysis["threats_detected"]),
            "query": threat_analysis.get("query", "")
        })
        
        # Keep only last 100 entries
        if len(self.threat_history[identifier]) > 100:
            self.threat_history[identifier] = self.threat_history[identifier][-100:]
        
        # Update cumulative risk score
        recent_requests = self.threat_history[identifier][-10:]  # Last 10 requests
        if recent_requests:
            avg_risk = sum(req["risk_score"] for req in recent_requests) / len(recent_requests)
            self.risk_scores[identifier] = avg_risk

class SecurityManager:
    """Main security management system"""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter()
        self.threat_detector = ThreatDetector()
        self.security_events = []
        self.security_config = self._load_security_config()
    
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        
        return {
            "max_security_events": 1000,
            "security_event_retention_days": 30,
            "auto_block_threshold": 0.8,
            "monitoring_enabled": True
        }
    
    def validate_request(self, 
                        content: str,
                        content_type: str = "query",
                        user_context: Dict[str, Any] = None,
                        user_id: Optional[str] = None,
                        ip_address: Optional[str] = None,
                        request_metadata: Dict[str, Any] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Comprehensive request validation and security analysis"""
        
        # Input validation
        is_valid, message, validation_data = self.input_validator.validate_input(
            content, content_type, user_context
        )
        
        if not is_valid:
            self._record_security_event(
                "input_validation_failed",
                "high",
                user_id,
                ip_address,
                {"content": content[:100], "error": message}
            )
            return False, message, {}
        
        # Rate limiting
        identifier = user_id or ip_address or "anonymous"
        within_limits, rate_limit_info = self.rate_limiter.check_rate_limit(
            identifier, "default", len(content)
        )
        
        if not within_limits:
            self._record_security_event(
                "rate_limit_exceeded",
                "medium",
                user_id,
                ip_address,
                {"rate_limit_info": rate_limit_info.__dict__}
            )
            return False, "Rate limit exceeded. Please wait before making another request.", {}
        
        # Threat detection
        threat_analysis = self.threat_detector.analyze_threat(
            content, user_id, ip_address, request_metadata
        )
        
        # Block if threat level is too high
        if threat_analysis["risk_score"] >= self.security_config["auto_block_threshold"]:
            self._record_security_event(
                "auto_blocked",
                "high",
                user_id,
                ip_address,
                {"threat_analysis": threat_analysis}
            )
            return False, "Request blocked due to security concerns.", {}
        
        # Record security event if medium or high risk
        if threat_analysis["risk_level"] in ["medium", "high"]:
            self._record_security_event(
                "threat_detected",
                threat_analysis["risk_level"],
                user_id,
                ip_address,
                {"threat_analysis": threat_analysis}
            )
        
        # Return success with security metadata
        security_metadata = {
            "validation_data": validation_data,
            "rate_limit_info": rate_limit_info.__dict__,
            "threat_analysis": threat_analysis,
            "security_validated": True
        }
        
        return True, "Request validated successfully", security_metadata
    
    def _record_security_event(self, 
                             event_type: str,
                             severity: str,
                             user_id: Optional[str],
                             ip_address: Optional[str],
                             details: Dict[str, Any]):
        """Record a security event"""
        
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            action_taken="logged"
        )
        
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > self.security_config["max_security_events"]:
            self.security_events = self.security_events[-self.security_config["max_security_events"]:]
        
        # Log security event
        logger.warning(f"Security event: {event_type} - {severity} - User: {user_id} - IP: {ip_address}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary"""
        
        # Calculate event statistics
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in self.security_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
        
        # Get rate limiting statistics
        rate_limit_stats = {}
        for identifier in list(self.rate_limiter.request_history.keys())[:10]:  # Top 10
            rate_limit_stats[identifier] = self.rate_limiter.get_rate_limit_info(identifier).__dict__
        
        # Get threat detection statistics
        threat_stats = {
            "total_identifiers": len(self.threat_detector.threat_history),
            "high_risk_identifiers": len([id for id, score in self.threat_detector.risk_scores.items() if score > 0.7]),
            "blocked_ips": len(self.rate_limiter.blocked_ips)
        }
        
        return {
            "security_events": {
                "total": len(self.security_events),
                "by_type": dict(event_counts),
                "by_severity": dict(severity_counts)
            },
            "rate_limiting": {
                "active_identifiers": len(self.rate_limiter.request_history),
                "blocked_identifiers": len(self.rate_limiter.blocked_ips),
                "top_identifiers": rate_limit_stats
            },
            "threat_detection": threat_stats,
            "system_status": "operational"
        }

# Global security manager instance
security_manager = SecurityManager()

# Convenience functions
def validate_request(content: str, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate request with security checks"""
    return security_manager.validate_request(content, **kwargs)

def get_security_summary() -> Dict[str, Any]:
    """Get security system summary"""
    return security_manager.get_security_summary()

def check_rate_limit(identifier: str, **kwargs) -> Tuple[bool, RateLimitInfo]:
    """Check rate limits for an identifier"""
    return security_manager.rate_limiter.check_rate_limit(identifier, **kwargs) 