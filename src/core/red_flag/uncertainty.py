"""
Uncertainty Detection Module

Specialized module for detecting uncertainty in agent responses.
This is a critical component of the red-flagging system.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .patterns import DetectionPattern


@dataclass
class UncertaintyScore:
    """Score representing the level of uncertainty in a response"""
    score: float  # 0.0 (certain) to 1.0 (very uncertain)
    markers_found: List[str]
    confidence_level: str  # "high", "medium", "low", "very_low"
    details: Dict[str, any] = None
    
    def is_acceptable(self, threshold: float = 0.3) -> bool:
        """Check if uncertainty is below acceptable threshold"""
        return self.score < threshold
    
    def __str__(self) -> str:
        return f"UncertaintyScore({self.score:.2f}, {self.confidence_level})"


class UncertaintyDetector:
    """
    Detects and quantifies uncertainty in agent responses.
    
    Based on MAKER paper's approach to identifying uncertain or low-quality outputs.
    """
    
    def __init__(self):
        self.uncertainty_patterns = self._initialize_patterns()
        self.hedging_patterns = self._initialize_hedging_patterns()
        self.qualifier_patterns = self._initialize_qualifier_patterns()
    
    def _initialize_patterns(self) -> DetectionPattern:
        """Initialize uncertainty detection patterns"""
        return DetectionPattern(
            name="uncertainty",
            patterns=[
                # Direct uncertainty
                "i'm not sure",
                "i'm unsure",
                "not certain",
                "uncertain",
                "unclear",
                "don't know",
                "can't tell",
                "hard to say",
                
                # Probabilistic hedging
                "probably",
                "possibly",
                "perhaps",
                "maybe",
                "might",
                "could be",
                "may be",
                "might be",
                
                # Belief-based (weak confidence)
                "i think",
                "i believe",
                "i guess",
                "i assume",
                "i suppose",
                "seems like",
                "appears to",
                "looks like",
                
                # Tentative language
                "somewhat",
                "kind of",
                "sort of",
                "rather",
                "fairly",
                "relatively",
                
                # Confidence qualifiers
                "not confident",
                "low confidence",
                "not entirely sure",
                "not completely certain",
                "tentatively"
            ],
            description="Patterns indicating uncertainty",
            severity="high"
        )
    
    def _initialize_hedging_patterns(self) -> List[str]:
        """Initialize hedging language patterns"""
        return [
            "it seems",
            "it appears",
            "it looks like",
            "it might be",
            "it could be",
            "this may",
            "this might",
            "this could",
            "potentially",
            "conceivably",
            "theoretically"
        ]
    
    def _initialize_qualifier_patterns(self) -> List[str]:
        """Initialize qualifier patterns that weaken statements"""
        return [
            "usually",
            "generally",
            "typically",
            "often",
            "sometimes",
            "occasionally",
            "in most cases",
            "in some cases",
            "under certain conditions",
            "depending on"
        ]
    
    def detect(self, text: str) -> UncertaintyScore:
        """
        Detect and score uncertainty in text.
        
        Args:
            text: The text to analyze
            
        Returns:
            UncertaintyScore with detailed analysis
        """
        text_lower = text.lower()
        
        # Find all uncertainty markers
        uncertainty_markers = self.uncertainty_patterns.find_matches(text)
        
        # Find hedging language
        hedging_found = [h for h in self.hedging_patterns if h in text_lower]
        
        # Find qualifiers
        qualifiers_found = [q for q in self.qualifier_patterns if q in text_lower]
        
        # Calculate score
        all_markers = uncertainty_markers + hedging_found + qualifiers_found
        
        # Score calculation
        base_score = min(len(uncertainty_markers) * 0.3, 1.0)
        hedging_score = min(len(hedging_found) * 0.2, 0.5)
        qualifier_score = min(len(qualifiers_found) * 0.1, 0.3)
        
        total_score = min(base_score + hedging_score + qualifier_score, 1.0)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(total_score)
        
        return UncertaintyScore(
            score=total_score,
            markers_found=all_markers,
            confidence_level=confidence_level,
            details={
                "uncertainty_markers": uncertainty_markers,
                "hedging_language": hedging_found,
                "qualifiers": qualifiers_found,
                "text_length": len(text),
                "marker_density": len(all_markers) / max(len(text.split()), 1)
            }
        )
    
    def _determine_confidence_level(self, score: float) -> str:
        """Determine confidence level from score"""
        if score < 0.2:
            return "high"
        elif score < 0.4:
            return "medium"
        elif score < 0.7:
            return "low"
        else:
            return "very_low"
    
    def has_uncertainty(self, text: str, threshold: float = 0.3) -> bool:
        """
        Check if text has unacceptable uncertainty.
        
        Args:
            text: Text to check
            threshold: Maximum acceptable uncertainty score
            
        Returns:
            True if uncertainty exceeds threshold
        """
        score = self.detect(text)
        return not score.is_acceptable(threshold)
    
    def get_uncertainty_markers(self, text: str) -> List[str]:
        """Get list of all uncertainty markers found in text"""
        score = self.detect(text)
        return score.markers_found
    
    def analyze_by_section(self, text: str, section_delimiter: str = "\n\n") -> List[Tuple[str, UncertaintyScore]]:
        """
        Analyze uncertainty section by section.
        
        Args:
            text: Text to analyze
            section_delimiter: Delimiter to split sections
            
        Returns:
            List of (section, score) tuples
        """
        sections = text.split(section_delimiter)
        results = []
        
        for section in sections:
            if section.strip():
                score = self.detect(section)
                results.append((section, score))
        
        return results
    
    def get_most_uncertain_section(self, text: str, section_delimiter: str = "\n\n") -> Optional[Tuple[str, UncertaintyScore]]:
        """Find the section with highest uncertainty"""
        sections = self.analyze_by_section(text, section_delimiter)
        
        if not sections:
            return None
        
        return max(sections, key=lambda x: x[1].score)


class ConfidenceAnalyzer:
    """Analyzes confidence indicators in responses"""
    
    def __init__(self):
        self.high_confidence_markers = [
            "definitely",
            "certainly",
            "absolutely",
            "clearly",
            "obviously",
            "undoubtedly",
            "without doubt",
            "for sure",
            "guaranteed",
            "proven"
        ]
        
        self.low_confidence_markers = [
            "not sure",
            "uncertain",
            "unclear",
            "ambiguous",
            "questionable",
            "debatable",
            "tentative"
        ]
    
    def analyze(self, text: str) -> Dict[str, any]:
        """Analyze confidence indicators in text"""
        text_lower = text.lower()
        
        high_conf_found = [m for m in self.high_confidence_markers if m in text_lower]
        low_conf_found = [m for m in self.low_confidence_markers if m in text_lower]
        
        # Calculate confidence ratio
        total_markers = len(high_conf_found) + len(low_conf_found)
        confidence_ratio = len(high_conf_found) / max(total_markers, 1)
        
        return {
            "high_confidence_markers": high_conf_found,
            "low_confidence_markers": low_conf_found,
            "confidence_ratio": confidence_ratio,
            "overall_confidence": "high" if confidence_ratio > 0.6 else "medium" if confidence_ratio > 0.3 else "low"
        }
    
    def has_low_confidence(self, text: str) -> bool:
        """Check if text shows low confidence"""
        analysis = self.analyze(text)
        return analysis["overall_confidence"] == "low"
