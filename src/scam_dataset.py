"""
SYNTHETIC SCAM DATASET - Industry-Grade
India-specific + global scam patterns for training and testing.

This is THE competitive edge: "We combined global scam datasets with 
India-specific curated data and synthetic multi-turn scam conversations."
"""

# ============================================================
# CATEGORY 1: UPI / FINANCIAL SCAMS (India-Specific)
# ============================================================

UPI_SCAM_MESSAGES = [
    # Accidental Transfer Scam
    {"text": "Sir, galti se 5000 aapke GPay pe bhej diya. Wapas kar do pls.", "scam_type": "UPI_REVERSAL", "language": "hinglish", "urgency": 0.6},
    {"text": "Hello, I accidentally sent Rs 10,000 to your PhonePe. Please refund to 9876543210@ybl", "scam_type": "UPI_REVERSAL", "language": "english", "urgency": 0.5},
    {"text": "Bhai galti ho gayi, 2000 bhej diya tumhare number pe. Return karo na urgent.", "scam_type": "UPI_REVERSAL", "language": "hinglish", "urgency": 0.7},
    
    # KYC Expiry Scam
    {"text": "Dear Customer, ur Paytm KYC expired. Call 98765xxxxx immediately or a/c block.", "scam_type": "KYC_SCAM", "language": "english", "urgency": 0.9},
    {"text": "ALERT: Your PhonePe KYC is pending. Complete within 24 hours or wallet will be suspended. Click: bit.ly/kyc-update", "scam_type": "KYC_SCAM", "language": "english", "urgency": 0.9},
    {"text": "SBI Alert: Aapka KYC expire ho gaya hai. Turant 1800-xxx-xxxx pe call karein ya account band ho jayega.", "scam_type": "KYC_SCAM", "language": "hinglish", "urgency": 0.95},
    
    # Fake Cashback/Reward
    {"text": "Congratulations! You've earned Rs 5000 cashback. Claim now: paytm-rewards.xyz/claim", "scam_type": "REWARD_SCAM", "language": "english", "urgency": 0.4},
    {"text": "GPay Special Offer! Scratch card worth ₹1000 waiting. Send ₹1 to activate: reward@gpay", "scam_type": "REWARD_SCAM", "language": "english", "urgency": 0.5},
    
    # OTP/Pin Theft
    {"text": "This is SBI helpdesk. We detected suspicious activity. Share your OTP for verification.", "scam_type": "OTP_THEFT", "language": "english", "urgency": 0.85},
    {"text": "Your account will be blocked in 30 minutes. Send OTP received on your phone to prevent.", "scam_type": "OTP_THEFT", "language": "english", "urgency": 0.95},
    {"text": "Sir aapka debit card block hone wala hai. Abhi OTP share karo nahi to aaj raat ko band ho jayega.", "scam_type": "OTP_THEFT", "language": "hinglish", "urgency": 0.9},
]

# ============================================================
# CATEGORY 2: SOCIAL ENGINEERING (Psychological Manipulation)
# ============================================================

SOCIAL_ENGINEERING_MESSAGES = [
    # Family Impersonation
    {"text": "Hi dad, I lost my phone. This is my friend's number +91-9876543210. I need 20k urgent for a new phone. Transfer to UPI: ravil@oksbi.", "scam_type": "FAMILY_IMPERSONATION", "language": "english", "urgency": 0.8},
    {"text": "Papa mera phone gir gaya toot gaya. Ye dost ka number hai. 15000 bhejo na urgent. UPI: suresh@paytm", "scam_type": "FAMILY_IMPERSONATION", "language": "hinglish", "urgency": 0.85},
    {"text": "Mom, I'm stuck without money. Can't access my bank. Please send 25000 to this account: 12345678901234, IFSC: SBIN0001234", "scam_type": "FAMILY_IMPERSONATION", "language": "english", "urgency": 0.9},
    
    # Authority Impersonation
    {"text": "This is Inspector Sharma from Cyber Crime Division. Your Aadhaar has been used in money laundering. Cooperate or face arrest.", "scam_type": "AUTHORITY_SCAM", "language": "english", "urgency": 0.95},
    {"text": "TRAI notice: Your mobile number will be disconnected within 2 hours due to illegal activity. Press 1 to speak to officer.", "scam_type": "AUTHORITY_SCAM", "language": "english", "urgency": 0.9},
    {"text": "Income Tax Department: Aapke PAN card pe suspicious transaction detect hua hai. Fine bharna padega.", "scam_type": "AUTHORITY_SCAM", "language": "hinglish", "urgency": 0.85},
    
    # Romance/Trust Scam
    {"text": "I've been talking to you for months and I trust you completely. Can you help me with a small emergency? Just Rs 50,000.", "scam_type": "ROMANCE_SCAM", "language": "english", "urgency": 0.6},
    {"text": "Darling, I'm stuck at customs. They need clearance fee of $500. Can you wire it? I'll pay back double when I arrive.", "scam_type": "ROMANCE_SCAM", "language": "english", "urgency": 0.7},
]

# ============================================================
# CATEGORY 3: PHISHING & LINK SCAMS
# ============================================================

PHISHING_MESSAGES = [
    {"text": "Your Amazon order #AMZ-9283 has been placed. If not authorized, cancel here: amazon-verify.sus-domain.com/cancel", "scam_type": "PHISHING", "language": "english", "urgency": 0.7},
    {"text": "Netflix account suspended. Update payment at netflix-billing.xyz/update to continue service.", "scam_type": "PHISHING", "language": "english", "urgency": 0.6},
    {"text": "SBI Internet Banking: Your account locked. Verify identity at sbi-netbanking.fake-link.com", "scam_type": "PHISHING", "language": "english", "urgency": 0.85},
    {"text": "WhatsApp: Your account will be deactivated. Verify: whatsapp-verify.scam.in/verify?user=victim", "scam_type": "PHISHING", "language": "english", "urgency": 0.8},
]

# ============================================================
# CATEGORY 4: LOTTERY / PRIZE SCAMS
# ============================================================

LOTTERY_MESSAGES = [
    {"text": "Badhai ho! Aapne Jeeto India contest me Tata Safari jeeti hai. Registration fee 5000 bhejein.", "scam_type": "LOTTERY_SCAM", "language": "hinglish", "urgency": 0.4},
    {"text": "CONGRATULATIONS! You have won $1,000,000 in the Microsoft International Lottery. Send processing fee of $100.", "scam_type": "LOTTERY_SCAM", "language": "english", "urgency": 0.3},
    {"text": "KBC Winner! Amitabh Bachchan ne aapka number select kiya hai. ₹25 lakh jeetne ke liye 9xxx pe call karein.", "scam_type": "LOTTERY_SCAM", "language": "hinglish", "urgency": 0.5},
    {"text": "You are selected for a Flipkart Big Billion lucky draw. Claim your iPhone 15 Pro: flipkart-prize.com", "scam_type": "LOTTERY_SCAM", "language": "english", "urgency": 0.4},
]

# ============================================================
# CATEGORY 5: JOB / INVESTMENT SCAMS
# ============================================================

JOB_INVESTMENT_MESSAGES = [
    {"text": "Work from home, earn ₹50,000/month. Just invest ₹5000 to start. WhatsApp 98xxx for details.", "scam_type": "JOB_SCAM", "language": "english", "urgency": 0.4},
    {"text": "Crypto investment opportunity! Guaranteed 300% returns in 30 days. Minimum deposit: 0.5 BTC to 0x742d35Cc6634C0532925a3b844Bc9e7595f2bD38", "scam_type": "INVESTMENT_SCAM", "language": "english", "urgency": 0.5},
    {"text": "Amazon/Flipkart product review job. Earn 2000-5000 daily. Deposit 1000 to start. UPI: taskmanager@oksbi", "scam_type": "JOB_SCAM", "language": "english", "urgency": 0.4},
]

# ============================================================
# CATEGORY 6: LEGITIMATE MESSAGES (False Positive Control)
# ============================================================

LEGITIMATE_MESSAGES = [
    {"text": "Hi, your food order from Swiggy will arrive in 20 minutes.", "scam_type": "LEGITIMATE", "language": "english", "urgency": 0.0},
    {"text": "Your Amazon package has been delivered. Thank you for shopping with us!", "scam_type": "LEGITIMATE", "language": "english", "urgency": 0.0},
    {"text": "Reminder: Your SIP of ₹5000 will be debited tomorrow. - HDFC Mutual Fund", "scam_type": "LEGITIMATE", "language": "english", "urgency": 0.1},
    {"text": "Bhai aaj shaam ko match dekhne chalein? India vs Australia hai.", "scam_type": "LEGITIMATE", "language": "hinglish", "urgency": 0.0},
    {"text": "Meeting rescheduled to 3 PM. Please join the Zoom link shared earlier.", "scam_type": "LEGITIMATE", "language": "english", "urgency": 0.2},
    {"text": "Happy Birthday! Wishing you a wonderful year ahead! 🎂", "scam_type": "LEGITIMATE", "language": "english", "urgency": 0.0},
]

# ============================================================
# CATEGORY 7: MULTI-TURN SCAM CONVERSATIONS (GOLD!)
# ============================================================

MULTI_TURN_SCAM_SCRIPTS = [
    {
        "scam_type": "UPI_REVERSAL",
        "description": "Classic accidental transfer scam with escalation",
        "turns": [
            {"sender": "scammer", "text": "Hello sir, I accidentally sent you ₹5000 on Google Pay. Can you please return it?", "phase": "INTRODUCTION"},
            {"sender": "agent", "text": "Oh really? Let me check my account...", "phase": "TRUST_BUILDING"},
            {"sender": "scammer", "text": "Yes sir, please check quickly. I need it urgently for my mother's medicine.", "phase": "URGENCY_CREATION"},
            {"sender": "agent", "text": "I don't see any credit. Are you sure you sent to the right number?", "phase": "TRUST_BUILDING"},
            {"sender": "scammer", "text": "Sir please, I am in hospital. My mother is very sick. Just send ₹5000 to this UPI: hospital_bill@paytm", "phase": "MONEY_REQUEST"},
            {"sender": "agent", "text": "Oh no, I hope she's okay. But I really don't see any amount credited...", "phase": "INFO_EXTRACTION"},
            {"sender": "scammer", "text": "SIR IT WILL REFLECT IN 24 HOURS! PLEASE SEND NOW OR MY MOTHER WILL DIE!", "phase": "ESCALATION"},
        ]
    },
    {
        "scam_type": "AUTHORITY_IMPERSONATION",
        "description": "Police/CBI impersonation with arrest threat",
        "turns": [
            {"sender": "scammer", "text": "This is CBI Officer Rajesh Kumar, badge number 4521. We have a warrant against your Aadhaar number.", "phase": "INTRODUCTION"},
            {"sender": "agent", "text": "What?! CBI? What did I do?", "phase": "TRUST_BUILDING"},
            {"sender": "scammer", "text": "Your Aadhaar has been linked to a money laundering case. ₹80 lakhs of black money was traced to your account.", "phase": "URGENCY_CREATION"},
            {"sender": "agent", "text": "But I'm just a retired teacher! I don't have that kind of money!", "phase": "URGENCY_CREATION"},
            {"sender": "scammer", "text": "That's what every criminal says. To avoid arrest, you must deposit ₹50,000 as security in our RBI verification account.", "phase": "MONEY_REQUEST"},
            {"sender": "agent", "text": "I'm so scared... which account should I transfer to?", "phase": "INFO_EXTRACTION"},
            {"sender": "scammer", "text": "Transfer immediately to account 98765432109876 IFSC PUNB0123456 or we will send police to your home TODAY.", "phase": "ESCALATION"},
        ]
    },
    {
        "scam_type": "KYC_SCAM",
        "description": "Paytm KYC scam with credential harvesting",
        "turns": [
            {"sender": "scammer", "text": "This is Paytm customer support. Your KYC verification has expired and your wallet will be blocked in 2 hours.", "phase": "INTRODUCTION"},
            {"sender": "agent", "text": "Oh no! I use Paytm for everything. What should I do?", "phase": "TRUST_BUILDING"},
            {"sender": "scammer", "text": "Don't worry, I'll help you complete KYC right now over the phone. First, please share your registered mobile number.", "phase": "INFO_EXTRACTION"},
            {"sender": "agent", "text": "It's 91-9999... wait, is this really Paytm? The number looks different.", "phase": "INFO_EXTRACTION"},
            {"sender": "scammer", "text": "Sir, this is the official Paytm helpline. For verification, please share the OTP you just received.", "phase": "INFO_EXTRACTION"},
            {"sender": "agent", "text": "I got a code... but my grandson says I shouldn't share OTP with anyone.", "phase": "INFO_EXTRACTION"},
            {"sender": "scammer", "text": "Your grandson is wrong! This is OFFICIAL verification. Share OTP in next 2 minutes or your ₹15,000 balance will be LOST FOREVER!", "phase": "ESCALATION"},
        ]
    },
]

# ============================================================
# CATEGORY 8: MANIPULATION PATTERNS (Psychological)
# ============================================================

MANIPULATION_PATTERNS = {
    "urgency_escalation": [
        "Send now", "immediately", "within 2 hours", "right now",
        "hurry up", "time is running out", "last chance", "deadline",
        "today only", "expires in", "before midnight"
    ],
    "authority_pressure": [
        "I am from the bank", "police department", "government official",
        "RBI authorized", "court order", "legal action", "CBI",
        "income tax", "cyber crime", "TRAI", "telecom authority"
    ],
    "fear_induction": [
        "your account will be blocked", "arrested", "jail",
        "legal action", "penalty", "fine", "suspended",
        "terminated", "blacklisted", "police complaint"
    ],
    "reward_lure": [
        "won a prize", "lottery winner", "cashback", "reward",
        "free gift", "bonus", "guaranteed returns", "special offer",
        "selected", "lucky draw", "congratulations"
    ],
    "trust_building": [
        "official verification", "for your safety", "secure process",
        "authorized", "certified", "government approved", "RBI guidelines",
        "customer protection", "verified", "standard procedure"
    ],
    "emotional_manipulation": [
        "my mother is sick", "hospital emergency", "accident",
        "stuck abroad", "lost everything", "crying", "please help",
        "I beg you", "life depends on it", "dying"
    ]
}

# ============================================================
# CATEGORY 9: SCAM SIGNATURES (For Vector Matching)
# ============================================================

SCAM_SIGNATURES = {
    "UPI_REVERSAL": {
        "signature_phrases": ["accidentally sent", "galti se bhej diya", "wapas kar do", "return the money", "wrong transfer"],
        "entities_expected": ["upi_id", "amount"],
        "risk_level": "HIGH"
    },
    "KYC_SCAM": {
        "signature_phrases": ["KYC expired", "KYC pending", "account block", "wallet suspended", "verification required"],
        "entities_expected": ["phone_number", "link"],
        "risk_level": "HIGH"
    },
    "OTP_THEFT": {
        "signature_phrases": ["share OTP", "send code", "verification code", "OTP for security", "confirm with OTP"],
        "entities_expected": ["otp_request"],
        "risk_level": "CRITICAL"
    },
    "AUTHORITY_SCAM": {
        "signature_phrases": ["police", "CBI", "arrest warrant", "money laundering", "Aadhaar fraud", "court order"],
        "entities_expected": ["authority_name", "threat"],
        "risk_level": "CRITICAL"
    },
    "LOTTERY_SCAM": {
        "signature_phrases": ["won lottery", "prize winner", "lucky draw", "claim prize", "registration fee"],
        "entities_expected": ["prize_name", "fee_amount"],
        "risk_level": "MEDIUM"
    },
    "FAMILY_IMPERSONATION": {
        "signature_phrases": ["lost my phone", "friend's number", "need money urgent", "stuck without cash"],
        "entities_expected": ["phone_number", "amount", "upi_id"],
        "risk_level": "HIGH"
    },
    "PHISHING": {
        "signature_phrases": ["click here", "verify identity", "update payment", "account locked", "suspended"],
        "entities_expected": ["phishing_url"],
        "risk_level": "HIGH"
    },
    "JOB_SCAM": {
        "signature_phrases": ["work from home", "earn daily", "part time job", "investment required", "guaranteed income"],
        "entities_expected": ["amount", "contact"],
        "risk_level": "MEDIUM"
    },
    "INVESTMENT_SCAM": {
        "signature_phrases": ["guaranteed returns", "300% profit", "crypto opportunity", "minimum deposit", "double your money"],
        "entities_expected": ["crypto_address", "amount"],
        "risk_level": "HIGH"
    },
    "ROMANCE_SCAM": {
        "signature_phrases": ["trust you", "emergency help", "stuck at customs", "wire money", "pay back double"],
        "entities_expected": ["amount", "account"],
        "risk_level": "MEDIUM"
    }
}


def get_all_scam_messages():
    """Get all scam messages for training/testing"""
    all_msgs = []
    all_msgs.extend(UPI_SCAM_MESSAGES)
    all_msgs.extend(SOCIAL_ENGINEERING_MESSAGES)
    all_msgs.extend(PHISHING_MESSAGES)
    all_msgs.extend(LOTTERY_MESSAGES)
    all_msgs.extend(JOB_INVESTMENT_MESSAGES)
    return all_msgs


def get_all_messages_with_labels():
    """Get all messages (scam + legitimate) with labels for training"""
    messages = []
    
    for msg in get_all_scam_messages():
        messages.append({
            "text": msg["text"],
            "is_scam": True,
            "scam_type": msg["scam_type"],
            "language": msg.get("language", "english"),
            "urgency": msg.get("urgency", 0.5)
        })
    
    for msg in LEGITIMATE_MESSAGES:
        messages.append({
            "text": msg["text"],
            "is_scam": False,
            "scam_type": "LEGITIMATE",
            "language": msg.get("language", "english"),
            "urgency": msg.get("urgency", 0.0)
        })
    
    return messages


def get_scam_type_from_text(text: str) -> str:
    """Classify scam type using signature matching"""
    text_lower = text.lower()
    best_match = "GENERIC_SCAM"
    best_score = 0
    
    for scam_type, sig in SCAM_SIGNATURES.items():
        score = sum(1 for phrase in sig["signature_phrases"] if phrase.lower() in text_lower)
        if score > best_score:
            best_score = score
            best_match = scam_type
    
    return best_match if best_score > 0 else "GENERIC_SCAM"
