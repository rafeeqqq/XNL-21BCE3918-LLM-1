from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
import time
import logging
import random
import uuid

# Import authentication utilities
from api.routes.auth import get_current_active_user, User

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

# Models
class RegulationType(str, Enum):
    SEC = "sec"
    GDPR = "gdpr"
    KYC = "kyc"
    AML = "aml"
    MiFID = "mifid"
    FINRA = "finra"
    BASEL = "basel"

class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REQUIRES_ACTION = "requires_action"
    EXEMPT = "exempt"

class DocumentType(str, Enum):
    TRADE_CONFIRMATION = "trade_confirmation"
    ACCOUNT_STATEMENT = "account_statement"
    PROSPECTUS = "prospectus"
    DISCLOSURE = "disclosure"
    MARKETING_MATERIAL = "marketing_material"
    FINANCIAL_REPORT = "financial_report"
    REGULATORY_FILING = "regulatory_filing"

class ComplianceCheck(BaseModel):
    id: str
    document_id: str
    document_type: DocumentType
    regulations: List[RegulationType]
    status: ComplianceStatus
    issues: List[str] = []
    recommendations: List[str] = []
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class AuditLogEntry(BaseModel):
    id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    timestamp: datetime

class KYCStatus(str, Enum):
    VERIFIED = "verified"
    PENDING = "pending"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REQUIRES_UPDATE = "requires_update"

class KYCVerification(BaseModel):
    id: str
    user_id: str
    status: KYCStatus
    verification_level: int  # 1, 2, 3 for basic, intermediate, advanced
    documents_provided: List[str]
    verification_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class RegulatoryReport(BaseModel):
    id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    submission_deadline: datetime
    status: str
    submitted_at: Optional[datetime] = None
    submitted_by: Optional[str] = None
    report_url: Optional[str] = None
    notes: Optional[str] = None

class RiskFlag(BaseModel):
    id: str
    user_id: Optional[str] = None
    account_id: Optional[str] = None
    transaction_id: Optional[str] = None
    flag_type: str
    severity: str  # low, medium, high, critical
    description: str
    status: str  # open, under_review, resolved, false_positive
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None

# Mock data generators
def generate_mock_compliance_check(document_id: str, document_type: DocumentType, regulations: List[RegulationType]) -> ComplianceCheck:
    """Generate mock compliance check result for testing"""
    # Randomly determine compliance status
    status_weights = {
        ComplianceStatus.COMPLIANT: 0.7,
        ComplianceStatus.NON_COMPLIANT: 0.1,
        ComplianceStatus.PENDING_REVIEW: 0.1,
        ComplianceStatus.REQUIRES_ACTION: 0.08,
        ComplianceStatus.EXEMPT: 0.02
    }
    
    status = random.choices(
        list(status_weights.keys()),
        weights=list(status_weights.values()),
        k=1
    )[0]
    
    issues = []
    recommendations = []
    
    # Generate issues and recommendations based on status and regulations
    if status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.REQUIRES_ACTION]:
        if RegulationType.SEC in regulations:
            issues.append("Insufficient risk disclosure in section 4.2")
            recommendations.append("Enhance risk disclosure to comply with SEC Rule 10b-5")
        
        if RegulationType.GDPR in regulations:
            issues.append("Missing explicit consent for data processing")
            recommendations.append("Add GDPR-compliant consent mechanism")
        
        if RegulationType.KYC in regulations:
            issues.append("Incomplete customer identification information")
            recommendations.append("Collect additional verification documents")
        
        if RegulationType.AML in regulations:
            issues.append("Unusual transaction pattern detected")
            recommendations.append("Conduct enhanced due diligence")
        
        if RegulationType.MiFID in regulations:
            issues.append("Product suitability assessment incomplete")
            recommendations.append("Complete client suitability assessment")
        
        if RegulationType.FINRA in regulations:
            issues.append("Inadequate disclosure of fees and commissions")
            recommendations.append("Revise fee disclosure to comply with FINRA Rule 2210")
        
        if RegulationType.BASEL in regulations:
            issues.append("Insufficient capital allocation for risk exposure")
            recommendations.append("Adjust capital reserves to meet Basel III requirements")
    
    # For PENDING_REVIEW, add some potential issues
    if status == ComplianceStatus.PENDING_REVIEW:
        issues.append("Awaiting expert review of complex regulatory requirements")
        recommendations.append("Schedule review with compliance officer")
    
    return ComplianceCheck(
        id=f"check-{uuid.uuid4()}",
        document_id=document_id,
        document_type=document_type,
        regulations=regulations,
        status=status,
        issues=issues,
        recommendations=recommendations,
        reviewed_by=None if status == ComplianceStatus.PENDING_REVIEW else "compliance-officer",
        reviewed_at=None if status == ComplianceStatus.PENDING_REVIEW else datetime.utcnow() - timedelta(days=random.randint(1, 10)),
        created_at=datetime.utcnow() - timedelta(days=random.randint(10, 30)),
        updated_at=datetime.utcnow() - timedelta(days=random.randint(0, 5))
    )

def generate_mock_audit_log(user_id: str, count: int = 10) -> List[AuditLogEntry]:
    """Generate mock audit log entries for testing"""
    entries = []
    
    actions = ["create", "read", "update", "delete", "approve", "reject", "submit", "download", "login", "logout"]
    resource_types = ["user", "account", "trade", "document", "report", "compliance_check", "kyc_verification"]
    
    for i in range(count):
        action = random.choice(actions)
        resource_type = random.choice(resource_types)
        
        # Generate plausible details based on action and resource type
        details = {}
        if action == "create":
            details["name"] = f"New {resource_type}"
        elif action == "update":
            details["fields_changed"] = random.randint(1, 5)
        elif action == "delete":
            details["reason"] = "Obsolete"
        elif action in ["approve", "reject"]:
            details["reason"] = "Compliance review"
            details["notes"] = f"Standard {action} process"
        
        entries.append(AuditLogEntry(
            id=f"audit-{uuid.uuid4()}",
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=f"{resource_type}-{random.randint(1000, 9999)}",
            details=details,
            ip_address=f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            timestamp=datetime.utcnow() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        ))
    
    # Sort by timestamp (newest first)
    entries.sort(key=lambda x: x.timestamp, reverse=True)
    
    return entries

def generate_mock_kyc_verification(user_id: str) -> KYCVerification:
    """Generate mock KYC verification for testing"""
    # Randomly determine KYC status
    status_weights = {
        KYCStatus.VERIFIED: 0.7,
        KYCStatus.PENDING: 0.15,
        KYCStatus.REJECTED: 0.05,
        KYCStatus.EXPIRED: 0.05,
        KYCStatus.REQUIRES_UPDATE: 0.05
    }
    
    status = random.choices(
        list(status_weights.keys()),
        weights=list(status_weights.values()),
        k=1
    )[0]
    
    # Determine verification level
    verification_level = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)[0]
    
    # Generate documents based on verification level
    documents = []
    if verification_level >= 1:
        documents.append("Government ID")
    if verification_level >= 2:
        documents.append("Proof of Address")
    if verification_level >= 3:
        documents.append("Source of Funds")
        documents.extend(["Bank Statement", "Tax Return"])
    
    # Set verification and expiry dates based on status
    verification_date = None
    expiry_date = None
    notes = None
    
    if status == KYCStatus.VERIFIED:
        verification_date = datetime.utcnow() - timedelta(days=random.randint(1, 365))
        expiry_date = verification_date + timedelta(days=365)
    elif status == KYCStatus.EXPIRED:
        verification_date = datetime.utcnow() - timedelta(days=random.randint(366, 730))
        expiry_date = verification_date + timedelta(days=365)
        notes = "Verification expired. Please resubmit documents."
    elif status == KYCStatus.REJECTED:
        notes = "Document quality insufficient. Please resubmit clearer images."
    elif status == KYCStatus.REQUIRES_UPDATE:
        verification_date = datetime.utcnow() - timedelta(days=random.randint(180, 364))
        expiry_date = verification_date + timedelta(days=365)
        notes = "Additional documents required due to account activity."
    
    return KYCVerification(
        id=f"kyc-{uuid.uuid4()}",
        user_id=user_id,
        status=status,
        verification_level=verification_level,
        documents_provided=documents,
        verification_date=verification_date,
        expiry_date=expiry_date,
        notes=notes,
        created_at=datetime.utcnow() - timedelta(days=random.randint(1, 730)),
        updated_at=datetime.utcnow() - timedelta(days=random.randint(0, 30))
    )

def generate_mock_regulatory_reports(count: int = 5) -> List[RegulatoryReport]:
    """Generate mock regulatory reports for testing"""
    reports = []
    
    report_types = [
        "Suspicious Activity Report (SAR)",
        "Currency Transaction Report (CTR)",
        "Form ADV",
        "Form 13F",
        "Form 10-K",
        "Form 10-Q",
        "FINRA Rule 4530",
        "MiFID II Transaction Report"
    ]
    
    statuses = ["draft", "pending_submission", "submitted", "accepted", "rejected", "requires_amendment"]
    
    for i in range(count):
        report_type = random.choice(report_types)
        
        # Generate realistic reporting periods
        period_end = datetime.utcnow() - timedelta(days=random.randint(1, 180))
        
        # Different report types have different reporting periods
        if "10-K" in report_type:
            period_start = period_end - timedelta(days=365)
        elif "10-Q" in report_type:
            period_start = period_end - timedelta(days=90)
        else:
            period_start = period_end - timedelta(days=random.choice([30, 90, 180, 365]))
        
        # Set submission deadline
        submission_deadline = period_end + timedelta(days=random.randint(15, 45))
        
        status = random.choice(statuses)
        
        # Set submitted info based on status
        submitted_at = None
        submitted_by = None
        report_url = None
        notes = None
        
        if status in ["submitted", "accepted", "rejected", "requires_amendment"]:
            submitted_at = period_end + timedelta(days=random.randint(1, 15))
            submitted_by = "compliance-officer"
            report_url = f"https://example.com/reports/{report_type.lower().replace(' ', '-')}-{period_end.strftime('%Y-%m-%d')}"
            
            if status == "rejected":
                notes = "Report rejected due to incomplete information. Please revise and resubmit."
            elif status == "requires_amendment":
                notes = "Minor corrections needed in section 3.2."
        
        reports.append(RegulatoryReport(
            id=f"report-{uuid.uuid4()}",
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            submission_deadline=submission_deadline,
            status=status,
            submitted_at=submitted_at,
            submitted_by=submitted_by,
            report_url=report_url,
            notes=notes
        ))
    
    # Sort by submission deadline (soonest first)
    reports.sort(key=lambda x: x.submission_deadline)
    
    return reports

def generate_mock_risk_flags(count: int = 5) -> List[RiskFlag]:
    """Generate mock risk flags for testing"""
    flags = []
    
    flag_types = [
        "unusual_trading_pattern",
        "large_transaction",
        "suspicious_login_location",
        "potential_insider_trading",
        "market_manipulation_attempt",
        "kyc_verification_issue",
        "high_risk_jurisdiction",
        "politically_exposed_person",
        "sanctions_list_match",
        "negative_news"
    ]
    
    severities = ["low", "medium", "high", "critical"]
    statuses = ["open", "under_review", "resolved", "false_positive"]
    
    for i in range(count):
        flag_type = random.choice(flag_types)
        severity = random.choices(severities, weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]
        status = random.choice(statuses)
        
        # Generate description based on flag type
        description = ""
        if flag_type == "unusual_trading_pattern":
            description = "Multiple large trades executed within short time period"
        elif flag_type == "large_transaction":
            description = f"Transaction exceeding ${random.randint(50, 500)}k threshold"
        elif flag_type == "suspicious_login_location":
            description = "Login from unusual geographic location"
        elif flag_type == "potential_insider_trading":
            description = "Trading activity correlated with material non-public information"
        elif flag_type == "market_manipulation_attempt":
            description = "Pattern of trades consistent with 'pump and dump' scheme"
        elif flag_type == "kyc_verification_issue":
            description = "Inconsistent identification information provided"
        elif flag_type == "high_risk_jurisdiction":
            description = "Transaction involving high-risk jurisdiction"
        elif flag_type == "politically_exposed_person":
            description = "Account holder identified as politically exposed person"
        elif flag_type == "sanctions_list_match":
            description = "Partial name match with sanctions list"
        elif flag_type == "negative_news":
            description = "Negative news associated with account holder"
        
        # Set resolution info based on status
        resolved_at = None
        resolved_by = None
        resolution_notes = None
        
        if status in ["resolved", "false_positive"]:
            resolved_at = datetime.utcnow() - timedelta(days=random.randint(1, 30))
            resolved_by = "compliance-officer"
            
            if status == "resolved":
                resolution_notes = "Issue investigated and addressed per compliance protocol"
            else:
                resolution_notes = "Investigation determined this was a false positive"
        
        created_at = datetime.utcnow() - timedelta(days=random.randint(1, 90))
        updated_at = created_at
        
        if status != "open":
            updated_at = created_at + timedelta(days=random.randint(1, 10))
        
        flags.append(RiskFlag(
            id=f"flag-{uuid.uuid4()}",
            user_id=f"user-{random.randint(1000, 9999)}" if random.random() < 0.8 else None,
            account_id=f"account-{random.randint(1000, 9999)}" if random.random() < 0.8 else None,
            transaction_id=f"tx-{random.randint(1000, 9999)}" if random.random() < 0.5 else None,
            flag_type=flag_type,
            severity=severity,
            description=description,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            resolved_at=resolved_at,
            resolved_by=resolved_by,
            resolution_notes=resolution_notes
        ))
    
    # Sort by created_at (newest first)
    flags.sort(key=lambda x: x.created_at, reverse=True)
    
    return flags

# Routes
@router.post("/check", response_model=ComplianceCheck)
async def check_compliance(
    document_id: str,
    document_type: DocumentType,
    regulations: List[RegulationType],
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate compliance check result
    result = generate_mock_compliance_check(document_id, document_type, regulations)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Compliance check request: Document={document_id}, Type={document_type}, Latency: {latency_ms:.3f}ms")
    
    return result

@router.get("/audit-log", response_model=List[AuditLogEntry])
async def get_audit_log(
    limit: int = 50,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate audit log entries
    entries = generate_mock_audit_log(user_id or current_user.username, limit)
    
    # Apply filters
    if resource_type:
        entries = [e for e in entries if e.resource_type == resource_type]
    
    if resource_id:
        entries = [e for e in entries if e.resource_id == resource_id]
    
    if action:
        entries = [e for e in entries if e.action == action]
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Audit log request: User={user_id or current_user.username}, Latency: {latency_ms:.3f}ms")
    
    return entries[:limit]

@router.get("/kyc/{user_id}", response_model=KYCVerification)
async def get_kyc_verification(
    user_id: str,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate KYC verification
    verification = generate_mock_kyc_verification(user_id)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"KYC verification request: User={user_id}, Latency: {latency_ms:.3f}ms")
    
    return verification

@router.get("/regulatory-reports", response_model=List[RegulatoryReport])
async def get_regulatory_reports(
    limit: int = 10,
    status: Optional[str] = None,
    report_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate regulatory reports
    reports = generate_mock_regulatory_reports(limit)
    
    # Apply filters
    if status:
        reports = [r for r in reports if r.status == status]
    
    if report_type:
        reports = [r for r in reports if report_type.lower() in r.report_type.lower()]
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Regulatory reports request: Latency: {latency_ms:.3f}ms")
    
    return reports[:limit]

@router.get("/risk-flags", response_model=List[RiskFlag])
async def get_risk_flags(
    limit: int = 10,
    user_id: Optional[str] = None,
    account_id: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # Generate risk flags
    flags = generate_mock_risk_flags(limit)
    
    # Apply filters
    if user_id:
        flags = [f for f in flags if f.user_id == user_id]
    
    if account_id:
        flags = [f for f in flags if f.account_id == account_id]
    
    if severity:
        flags = [f for f in flags if f.severity == severity]
    
    if status:
        flags = [f for f in flags if f.status == status]
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Risk flags request: Latency: {latency_ms:.3f}ms")
    
    return flags[:limit]

@router.post("/blockchain-audit", response_model=Dict[str, Any])
async def create_blockchain_audit_record(
    record_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
):
    # Start timer for latency measurement
    start_time = time.time()
    
    # In a real system, this would create an immutable record on a blockchain
    # For this mock, we'll just return a simulated response
    
    # Generate a mock transaction hash
    tx_hash = "0x" + "".join(random.choices("0123456789abcdef", k=64))
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log the request
    logger.info(f"Blockchain audit record creation: Latency: {latency_ms:.3f}ms")
    
    return {
        "success": True,
        "transaction_hash": tx_hash,
        "block_number": random.randint(10000000, 20000000),
        "timestamp": datetime.utcnow().isoformat(),
        "network": "Hyperledger Fabric",
        "record_id": str(uuid.uuid4())
    }
