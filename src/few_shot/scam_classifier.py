"""
Few-Shot Learning for New Scam Type Detection
Using Prototypical Networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot classification
    """
    def __init__(self, embedding_dim=768, hidden_dim=256):
        super().__init__()
        
        # Projection network
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch, embedding_dim)
        Returns:
            projected: (batch, hidden_dim)
        """
        return self.projection(embeddings)

class FewShotScamClassifier:
    """
    Few-shot learning for detecting new scam types with minimal examples
    
    Meta-learning approach:
    1. Train on diverse scam types
    2. At test time, provide 5 examples of new scam type
    3. Classify new instances
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sentence encoder
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
            self.encoder.eval()
            self.available = True
        except:
            print("⚠️ Sentence transformer not available, using fallback")
            self.available = False
        
        # Prototypical network
        self.proto_net = PrototypicalNetwork(embedding_dim=384, hidden_dim=128).to(self.device)
        self.proto_net.eval()
        
        # Known scam type prototypes (learned during meta-training)
        self.known_prototypes = {}
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into embeddings
        
        Args:
            texts: list of strings
        Returns:
            embeddings: (len(texts), embedding_dim)
        """
        if not self.available:
            # Fallback: random embeddings
            return torch.randn(len(texts), 384).to(self.device)
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    def compute_prototypes(self, support_texts: List[str], support_labels: List[int]) -> Dict[int, torch.Tensor]:
        """
        Compute class prototypes from support set
        
        Args:
            support_texts: list of example texts
            support_labels: list of labels (0, 1, 2, ...)
        Returns:
            prototypes: dict mapping label -> prototype vector
        """
        # Encode support set
        embeddings = self.encode_text(support_texts)
        projected = self.proto_net(embeddings)
        
        # Compute prototypes (mean of each class)
        prototypes = {}
        unique_labels = set(support_labels)
        
        for label in unique_labels:
            mask = torch.tensor([l == label for l in support_labels]).to(self.device)
            class_embeddings = projected[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes[label] = prototype
        
        return prototypes
    
    def classify_with_prototypes(self, 
                                  query_texts: List[str],
                                  prototypes: Dict[int, torch.Tensor],
                                  label_names: Dict[int, str] = None) -> List[Tuple[int, float, str]]:
        """
        Classify query texts using prototypes
        
        Args:
            query_texts: texts to classify
            prototypes: class prototypes
            label_names: optional mapping of label -> name
        Returns:
            predictions: list of (label, confidence, name)
        """
        # Encode queries
        embeddings = self.encode_text(query_texts)
        projected = self.proto_net(embeddings)
        
        # Compute distances to prototypes
        predictions = []
        for query_emb in projected:
            distances = {}
            for label, prototype in prototypes.items():
                # Euclidean distance
                dist = torch.norm(query_emb - prototype).item()
                distances[label] = dist
            
            # Closest prototype
            pred_label = min(distances, key=distances.get)
            
            # Convert distance to confidence (inverse distance)
            all_dists = list(distances.values())
            min_dist = min(all_dists)
            confidence = 1.0 / (1.0 + min_dist)
            
            # Get name
            name = label_names.get(pred_label, f"Class_{pred_label}") if label_names else f"Class_{pred_label}"
            
            predictions.append((pred_label, confidence, name))
        
        return predictions
    
    def few_shot_classify(self,
                          support_examples: Dict[str, List[str]],
                          query_texts: List[str]) -> List[Dict]:
        """
        High-level few-shot classification
        
        Args:
            support_examples: {
                "UPI_SCAM": ["example1", "example2", ...],
                "PRIZE_SCAM": ["example1", "example2", ...],
                ...
            }
            query_texts: texts to classify
        
        Returns:
            predictions: list of dicts with classification results
        """
        # Prepare support set
        support_texts = []
        support_labels = []
        label_to_name = {}
        
        for label_id, (scam_type, examples) in enumerate(support_examples.items()):
            support_texts.extend(examples)
            support_labels.extend([label_id] * len(examples))
            label_to_name[label_id] = scam_type
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_texts, support_labels)
        
        # Classify queries
        predictions = self.classify_with_prototypes(query_texts, prototypes, label_to_name)
        
        # Format results
        results = []
        for i, (label, conf, name) in enumerate(predictions):
            results.append({
                "text": query_texts[i],
                "predicted_scam_type": name,
                "confidence": conf,
                "label_id": label
            })
        
        return results

# Example usage
def demo_few_shot():
    """
    Demonstrate few-shot learning with 5 examples per class
    """
    classifier = FewShotScamClassifier()
    
    # Support set: 5 examples of each scam type
    support_examples = {
        "UPI_SCAM": [
            "Send money to this UPI ID immediately",
            "Your payment failed, retry with this UPI",
            "Refund will be credited to your UPI account",
            "Verify your UPI by sending Rs 1",
            "Update your UPI details now"
        ],
        "PRIZE_SCAM": [
            "Congratulations! You won a lottery",
            "You are selected for a prize of Rs 10 lakh",
            "Claim your reward by clicking here",
            "You won a lucky draw contest",
            "Prize money waiting for you"
        ],
        "BANK_SCAM": [
            "Your account will be blocked today",
            "Urgent: Update your bank KYC",
            "Your debit card is suspended",
            "Bank account verification required",
            "Confirm your account details immediately"
        ]
    }
    
    # Query: new messages to classify
    query_texts = [
        "Your UPI payment is pending, send Rs 10 to confirm",  # UPI_SCAM
        "You won Rs 5 crore in a lottery",  # PRIZE_SCAM
        "Your bank account will be closed tomorrow"  # BANK_SCAM
    ]
    
    # Classify
    results = classifier.few_shot_classify(support_examples, query_texts)
    
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Predicted: {result['predicted_scam_type']} ({result['confidence']:.2f})")
        print()
    
    return results
