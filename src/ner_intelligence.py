"""
NER-based Intelligence Extraction using SpaCy
"""
import re
from typing import Dict, List

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


class NERIntelligence:
    def __init__(self):
        """Initialize SpaCy NER model"""
        self.nlp = None
        if not SPACY_AVAILABLE:
            print("⚠️ SpaCy not installed. Using regex-only extraction (faster!)")
            return
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            print("Falling back to regex-only extraction")
    
    def extract_intelligence(self, text: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Extract intelligence using NER + Regex
        Returns structured intel with confidence scores
        """
        intel = {
            "bankAccounts": [],
            "upiIds": [],
            "phishingLinks": [],
            "phoneNumbers": [],
            "suspiciousKeywords": [],
            "ifscCodes": [],
            "cryptoAddresses": [],
            "emailAddresses": []
        }
        
        confidences = {}
        
        # 1. Regex-based extraction (high confidence)
        regex_intel = self._extract_with_regex(text)
        for key, values in regex_intel.items():
            intel[key].extend(values)
            for val in values:
                confidences[val] = 0.9  # High confidence for regex matches
        
        # 2. NER-based extraction (variable confidence)
        if self.nlp:
            ner_intel = self._extract_with_ner(text)
            for key, items in ner_intel.items():
                for item, conf in items:
                    if conf >= confidence_threshold:
                        if item not in intel[key]:
                            intel[key].append(item)
                            confidences[item] = conf
        
        # 3. Keyword extraction
        keywords = self._extract_keywords(text)
        intel["suspiciousKeywords"].extend(keywords)
        
        # Deduplicate all lists
        for key in intel:
            if isinstance(intel[key], list):
                intel[key] = list(set(intel[key]))
        
        # Add confidence metadata
        intel["_confidence_scores"] = confidences
        
        return intel
    
    def _extract_with_regex(self, text: str) -> Dict:
        """Advanced regex-based extraction — GENERIC, works for any scam scenario"""
        intel = {
            "bankAccounts": [],
            "upiIds": [],
            "phishingLinks": [],
            "phoneNumbers": [],
            "ifscCodes": [],
            "cryptoAddresses": [],
            "emailAddresses": []
        }
        
        # 1. URLs/Phishing Links — capture FULL URLs with paths and query params
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        intel["phishingLinks"] = re.findall(url_pattern, text)
        
        # 2. UPI IDs — GENERIC: match ANY user@handle pattern
        # UPI format: anything@anything (NOT standard emails with .com/.org)
        # Examples: scammer.fraud@fakebank, cashback.scam@fakeupi, user@paytm
        upi_pattern = r'\b[a-zA-Z0-9._-]+@[a-zA-Z][a-zA-Z0-9._-]*[a-zA-Z]\b'
        all_at_patterns = re.findall(upi_pattern, text)
        
        # Separate into UPI IDs vs emails
        # If it has a proper TLD (.com, .org, .net, .in, .edu, etc.) → email
        # Otherwise → UPI ID
        email_tlds = ['.com', '.org', '.net', '.in', '.edu', '.gov', '.co.in', '.io', '.info', '.biz', '.xyz']
        for match in all_at_patterns:
            is_email = any(match.lower().endswith(tld) for tld in email_tlds)
            if is_email:
                intel["emailAddresses"].append(match)
            else:
                intel["upiIds"].append(match)
        
        # 3. Indian Phone Numbers — handle ALL formats including hyphens
        phone_patterns = [
            r'\+91[-\s]?\d{10}',         # +91-9876543210 or +91 9876543210
            r'\+91[-\s]?\d{5}[-\s]?\d{5}', # +91-98765-43210
            r'91\d{10}',                   # 919876543210
            r'\b0?[6-9]\d{9}\b'            # 9876543210 or 09876543210
        ]
        for pattern in phone_patterns:
            intel["phoneNumbers"].extend(re.findall(pattern, text))
        # Deduplicate
        intel["phoneNumbers"] = list(set(intel["phoneNumbers"]))
        
        # 4. Bank Account Numbers (9-18 digits)
        account_pattern = r'\b\d{9,18}\b'
        potential_accounts = re.findall(account_pattern, text)
        # Filter out phone numbers
        clean_phones = set()
        for p in intel["phoneNumbers"]:
            # Strip +91, 91, 0 prefix and hyphens/spaces to get raw 10 digits
            cleaned = re.sub(r'[\+\-\s]', '', p)
            if cleaned.startswith('91') and len(cleaned) > 10:
                cleaned = cleaned[2:]
            elif cleaned.startswith('0'):
                cleaned = cleaned[1:]
            clean_phones.add(cleaned)
        
        intel["bankAccounts"] = [
            acc for acc in potential_accounts 
            if acc not in clean_phones and not acc.startswith('20')
        ]
        
        # 5. IFSC Codes (format: ABCD0123456)
        ifsc_pattern = r'\b[A-Z]{4}0[A-Z0-9]{6}\b'
        intel["ifscCodes"] = re.findall(ifsc_pattern, text)
        
        # 6. Also extract emails that look like real emails (from URLs or text)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails_from_text = re.findall(email_pattern, text)
        for email in emails_from_text:
            if email not in intel["emailAddresses"] and email not in intel["upiIds"]:
                intel["emailAddresses"].append(email)
        
        # 7. Cryptocurrency addresses (Bitcoin, Ethereum)
        btc_pattern = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'
        eth_pattern = r'\b0x[a-fA-F0-9]{40}\b'
        intel["cryptoAddresses"].extend(re.findall(btc_pattern, text))
        intel["cryptoAddresses"].extend(re.findall(eth_pattern, text))
        
        return intel
    
    def _extract_with_ner(self, text: str) -> Dict:
        """NER-based extraction using SpaCy"""
        intel = {
            "bankAccounts": [],
            "phoneNumbers": [],
            "emailAddresses": []
        }
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            confidence = 0.7  # Base NER confidence
            
            if ent.label_ == "CARDINAL" and len(ent.text) >= 9:
                # Potential account number
                intel["bankAccounts"].append((ent.text, confidence))
            
            elif ent.label_ in ["PERSON", "ORG"]:
                # Sometimes names/orgs can be part of UPI or email
                pass  # Already handled by regex
        
        return intel
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract suspicious keywords"""
        keywords = [
            "urgent", "verify", "block", "suspend", "expire", "immediately",
            "click", "link", "confirm", "account", "password", "otp", "cvv",
            "kyc", "lottery", "prize", "congratulations", "winner",
            "bank", "payment", "transfer", "upi", "debit", "credit"
        ]
        
        text_lower = text.lower()
        found = [kw for kw in keywords if kw in text_lower]
        return list(set(found))
