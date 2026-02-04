"""
NER-based Intelligence Extraction using SpaCy
"""
import re
from typing import Dict, List
import spacy

class NERIntelligence:
    def __init__(self):
        """Initialize SpaCy NER model"""
        self.nlp = None
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
        """Advanced regex-based extraction"""
        intel = {
            "bankAccounts": [],
            "upiIds": [],
            "phishingLinks": [],
            "phoneNumbers": [],
            "ifscCodes": [],
            "cryptoAddresses": [],
            "emailAddresses": []
        }
        
        # URLs (phishing links)
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        intel["phishingLinks"] = re.findall(url_pattern, text)
        
        # UPI IDs (format: username@bankname)
        upi_pattern = r'\b[a-zA-Z0-9._-]+@[a-zA-Z]{3,}\b'
        potential_upi = re.findall(upi_pattern, text)
        # Filter to keep only valid UPI providers
        upi_providers = ['paytm', 'phonepe', 'gpay', 'googlepay', 'bhim', 'okaxis', 'oksbi', 'okicici', 'okhdfc', 'ybl', 'ibl', 'axl']
        intel["upiIds"] = [upi for upi in potential_upi if any(provider in upi.lower() for provider in upi_providers)]
        
        # Indian Phone Numbers
        phone_patterns = [
            r'\+91[-\s]?\d{10}',  # +91 format
            r'91\d{10}',  # 91 prefix
            r'\b[6-9]\d{9}\b'  # 10 digit starting with 6-9
        ]
        for pattern in phone_patterns:
            intel["phoneNumbers"].extend(re.findall(pattern, text))
        
        # Bank Account Numbers (9-18 digits, excluding obvious dates/years)
        account_pattern = r'\b\d{9,18}\b'
        potential_accounts = re.findall(account_pattern, text)
        # Filter out years, dates, phone numbers already found
        intel["bankAccounts"] = [
            acc for acc in potential_accounts 
            if len(acc) >= 10 and acc not in intel["phoneNumbers"] and not acc.startswith('20')
        ]
        
        # IFSC Codes (format: ABCD0123456)
        ifsc_pattern = r'\b[A-Z]{4}0[A-Z0-9]{6}\b'
        intel["ifscCodes"] = re.findall(ifsc_pattern, text)
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        # Exclude UPI IDs from emails
        intel["emailAddresses"] = [email for email in emails if email not in intel["upiIds"]]
        
        # Cryptocurrency addresses (Bitcoin, Ethereum)
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
