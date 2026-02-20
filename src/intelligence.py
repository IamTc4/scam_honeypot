import re

def extract_intel(text: str) -> dict:
    """
    Extracts structured intelligence from text using Regex.
    Returns a dictionary of found items.
    """
    intel = {
        "bankAccounts": [],
        "upiIds": [],
        "phishingLinks": [],
        "phoneNumbers": [],
        "suspiciousKeywords": []
    }
    
    # URL Pattern
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    intel["phishingLinks"] = re.findall(url_pattern, text)
    
    # UPI Pattern (simplified)
    upi_pattern = r'[\w\.\-_]+@[\w]+'
    potential_upi = re.findall(upi_pattern, text)
    # Filter out email-like false positives if needed, but UPIs look like emails.
    intel["upiIds"] = potential_upi
    
    # Phone Number Pattern (India specific mostly, but generic enough)
    phone_pattern = r'(?:\+91|91|0)?[6-9]\d{9}'
    intel["phoneNumbers"] = re.findall(phone_pattern, text)
    
    # Bank Account (Generic 9-18 digit numbers)
    # This is tricky as OTPs can also be numbers. We'll look for "acc" or "no" context in a real ML model.
    # For extraction, we grab long logical number sequences.
    account_pattern = r'\b\d{9,18}\b'
    intel["bankAccounts"] = re.findall(account_pattern, text)
    
    # Scam keywords
    keywords = ["urgent", "verify", "block", "suspend", "kyc", "expire"]
    intel["suspiciousKeywords"] = [kw for kw in keywords if kw in text.lower()]
    
    # Deduplicate
    for key in intel:
        if isinstance(intel[key], list):
             intel[key] = list(set(intel[key]))
             
    return intel
