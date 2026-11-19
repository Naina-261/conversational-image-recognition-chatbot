from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import logging

logger = logging.getLogger("bel_chatbot.context_manager")


class QuestionIntent(str, Enum):
    OBJECT_INVENTORY = "object_inventory"
    OBJECT_COUNT = "object_count"
    SPATIAL_LOCATION = "spatial_location"
    VISUAL_ATTRIBUTE = "visual_attribute"
    GENERAL = "general"


CLASS_SYNONYMS: Dict[str, List[str]] = {
    "car": ["car", "cars", "automobile", "automobiles", "sedan", "sedans", "auto", "vehicle", "vehicles"],
    "traffic light": ["traffic light", "traffic lights", "stoplight", "stoplights", "traffic signal", "traffic signals", "signal", "signals"],
    "truck": ["truck", "trucks", "pickup", "lorry", "lorries"],
    "bus": ["bus", "buses"],
    "person": ["person", "persons", "people", "pedestrian", "pedestrians", "human", "humans", "man", "men", "woman", "women"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes", "cyclist", "cyclists"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
    "airplane": ["airplane", "airplanes", "plane", "planes", "aircraft"],
    "tie": ["tie", "ties", "necktie", "neckties"],
    "train": ["train", "trains"],
    "boat": ["boat", "boats", "ship", "ships", "vessel", "vessels"],
    "dog": ["dog", "dogs", "puppy", "puppies", "canine"],
    "cat": ["cat", "cats", "kitten", "kittens", "feline"],
    "chair": ["chair", "chairs", "seat", "seats"],
    "bench": ["bench", "benches"],
    "fire hydrant": ["fire hydrant", "fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop sign", "stop signs"],
}

CATEGORY_GROUPS: Dict[str, List[str]] = {
    "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle"],
}

INVENTORY_TRIGGERS = [
    r"\bwhat objects?\b",
    r"\bwhich objects?\b",
    r"\bwhat.*detected\b",
    r"\bdetected objects?\b",
    r"\bwhat.*visible\b",
    r"\bwhat.*in (?:this|the) image\b",
    r"\bwhat do you see\b",
    r"\blist.*objects?\b",
    r"\bwhat categories\b",
    r"\bidentify.*objects?\b",
    r"\bwhat are the objects\b",
    r"\bwhat did you find\b",
    r"\bwhat.*found\b",
]

COUNT_TRIGGERS = [
    r"\bhow many\b",
    r"\bnumber of\b",
    r"\bcount\b",
    r"\btotal (?:number of )?objects?\b",
]

SPATIAL_TRIGGERS = [
    r"\bwhere (?:is|are|was|were)\b",
    r"\bwhere.*located\b",
    r"\bwhat is (?:on|in|at) the (?:left|right|center|middle|top|bottom)\b",
    r"\bwhich (?:side|part|sector|area|quadrant)\b",
    r"\blocation of\b",
    r"\bposition of\b",
    r"\b(?:parked|standing|driving|stopped|located|positioned|placed)\s+(?:on|in|at)\b",
    r"\b(?:on|in|at)\s+(?:the\s+)?(?:road|street|sidewalk|pavement|driveway|lawn|grass|ground|lane)\b",
    r"\b(?:road|street)\s+or\s+(?:the\s+)?(?:sidewalk|pavement|driveway)\b",
    r"\b(?:sidewalk|pavement)\s+or\s+(?:the\s+)?(?:road|street|driveway)\b",
]

ATTRIBUTE_TRIGGERS = [
    r"\bwhat color\b",
    r"\bwhich color\b",
    r"\bcolor of\b",
    r"\bwhat.*wearing\b",
    r"\bwhat.*doing\b",
    r"\bwhat.*look like\b",
    r"\bweather\b",
    r"\btime of day\b",
    r"\bday or night\b",
    r"\bsetting\b",
    r"\bbackground\b",
    r"\benvironment\b",
    r"\bmake\b",
    r"\bbrand\b",
    r"\bmodel\b",
    r"\bdescription\b",
    r"\bdescribe\b",
]


class DialogueTurn:
    """Represents a single conversational exchange turn."""
    def __init__(self, sender_type: str, content: str, metadata: Optional[dict] = None):
        self.sender_type = sender_type
        self.content = content
        self.metadata = metadata or {}


class ConversationalContextManager:
    """
    Multi-turn conversational context manager.
    Tracks dialogue turns, resolves pronoun coreferences, and grounds queries
    in visual and conversational history.
    """

    def __init__(self, max_history_window: int = 10):
        self.max_history_window = max_history_window

    def build_history_window(self, messages: List[Any]) -> List[DialogueTurn]:
        turns = []
        for msg in messages[-self.max_history_window:]:
            sender = getattr(msg, "sender_type", None) or (msg.get("sender_type") if isinstance(msg, dict) else None)
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            meta = getattr(msg, "metadata_payload", None) or getattr(msg, "metadata", {}) if not isinstance(msg, dict) else msg.get("metadata", {})
            if sender in ("user", "assistant"):
                turns.append(DialogueTurn(sender, content, meta))
        return turns

    @staticmethod
    def normalize_class_name(word: str) -> Optional[str]:
        w = word.lower().strip()
        for canonical, syns in CLASS_SYNONYMS.items():
            if w in syns:
                return canonical
        for canonical in CLASS_SYNONYMS.keys():
            if w == canonical or w == f"{canonical}s" or w == f"{canonical}es":
                return canonical
        return None

    @classmethod
    def match_target_class_or_group(cls, text: str) -> Tuple[Optional[str], Optional[List[str]]]:
        t = text.lower()
        for group_name, members in CATEGORY_GROUPS.items():
            pattern = r"\b" + group_name + r"\b"
            if re.search(pattern, t):
                return group_name, members

        for canonical, syns in CLASS_SYNONYMS.items():
            for syn in sorted(syns, key=len, reverse=True):
                if " " in syn and re.search(r"\b" + re.escape(syn) + r"\b", t):
                    return canonical, [canonical]

        words = re.findall(r"\b[a-zA-Z_-]+\b", t)
        for w in words:
            canon = cls.normalize_class_name(w)
            if canon:
                return canon, [canon]

        return None, None

    def resolve_pronouns_and_coreference(
        self,
        current_query: str,
        dialogue_history: List[DialogueTurn],
        active_detected_objects: List[dict],
    ) -> str:
        intent, resolved, _, _, _, _, _ = self.classify_and_resolve(
            current_query, dialogue_history, active_detected_objects
        )
        return resolved

    def classify_and_resolve(
        self,
        current_query: str,
        dialogue_history: List[DialogueTurn],
        active_detected_objects: List[dict],
    ) -> Tuple[QuestionIntent, str, Optional[str], Optional[List[str]], Optional[str], Optional[float], bool]:
        """
        Classifies intent, parses confidence threshold from query if present,
        and resolves pronouns/coreferences.
        Returns:
            (intent, resolved_query, target_class, matched_classes, spatial_target, explicit_threshold, is_follow_up)
        """
        q = current_query.strip()
        q_lower = q.lower()

        # 1. Parse explicit threshold in text if present
        explicit_threshold = None
        pct_match = re.search(r"(\d{1,2})\s*%\s*(?:confidence|conf)?", q_lower)
        if pct_match:
            explicit_threshold = float(pct_match.group(1)) / 100.0
        else:
            dec_match = re.search(r"(?:confidence|threshold)\s*(?:of|at|above|>=|>)?\s*(0\.\d+)", q_lower)
            if dec_match:
                explicit_threshold = float(dec_match.group(1))

        # Check for coreference pronouns
        pronouns = ["it", "they", "them", "that", "this", "these", "those"]
        words = re.findall(r"\b\w+\b", q_lower)
        has_pronoun = any(p in words for p in pronouns) or "that vehicle" in q_lower or "the vehicle" in q_lower
        is_follow_up = bool(dialogue_history) and (has_pronoun or any(w in words for w in ["also", "and", "too"]))

        referenced_class = None
        referenced_group = None
        if dialogue_history and has_pronoun:
            for turn in reversed(dialogue_history):
                if turn.metadata and turn.metadata.get("primary_subject"):
                    sub = turn.metadata["primary_subject"]
                    if sub == "detected objects":
                        referenced_class = "detected objects"
                        referenced_group = None
                        break
                    if sub in CLASS_SYNONYMS or sub in CATEGORY_GROUPS:
                        referenced_class = sub
                        referenced_group = CATEGORY_GROUPS.get(sub, [sub])
                        break
                cand_class, cand_group = self.match_target_class_or_group(turn.content)
                if cand_class:
                    referenced_class = cand_class
                    referenced_group = cand_group
                    break

        # 2. Classify Intent
        # A. Spatial Query
        spatial_match = None
        for direction in ["top-left", "top-right", "top-center", "bottom-left", "bottom-right", "bottom-center", "middle-left", "middle-right", "center", "middle", "left", "right", "top", "bottom"]:
            if re.search(r"\b" + direction + r"\b", q_lower):
                spatial_match = direction
                break

        is_spatial_question = any(re.search(pat, q_lower) for pat in SPATIAL_TRIGGERS) or (
            spatial_match and any(w in words for w in ["what", "which", "where", "show"])
        )

        if is_spatial_question:
            target_class, matched_group = self.match_target_class_or_group(q_lower)
            if not target_class and referenced_class:
                if referenced_class == "detected objects":
                    target_class = None
                    matched_group = None
                else:
                    target_class = referenced_class
                    matched_group = referenced_group
            resolved = q
            if target_class and has_pronoun:
                resolved = re.sub(r"\b(it|they|them|that|this)\b", f"the {target_class}", q, flags=re.IGNORECASE)
            return QuestionIntent.SPATIAL_LOCATION, resolved, target_class, matched_group, spatial_match, explicit_threshold, is_follow_up

        # B. Count Query
        is_count_question = any(re.search(pat, q_lower) for pat in COUNT_TRIGGERS)
        if is_count_question:
            target_class, matched_group = self.match_target_class_or_group(q_lower)
            if not target_class and referenced_class:
                target_class = referenced_class
                matched_group = referenced_group
            resolved = q
            if target_class and has_pronoun:
                resolved = re.sub(r"\b(it|they|them|that|this)\b", f"{target_class}", q, flags=re.IGNORECASE)
            return QuestionIntent.OBJECT_COUNT, resolved, target_class, matched_group, None, explicit_threshold, is_follow_up

        # C. Object Inventory / Identification Query
        is_inventory_question = any(re.search(pat, q_lower) for pat in INVENTORY_TRIGGERS)
        if is_inventory_question:
            return QuestionIntent.OBJECT_INVENTORY, q, None, None, None, explicit_threshold, is_follow_up

        # D. Visual Attribute Query (Color, Appearance, Weather, Action)
        is_attribute_question = any(re.search(pat, q_lower) for pat in ATTRIBUTE_TRIGGERS)
        if is_attribute_question:
            target_class, matched_group = self.match_target_class_or_group(q_lower)
            if not target_class and referenced_class:
                target_class = referenced_class
                matched_group = referenced_group
            resolved = q
            if target_class and has_pronoun:
                resolved = re.sub(r"\b(it|that|this)\b", f"the {target_class}", q, flags=re.IGNORECASE)
                resolved = re.sub(r"\b(they|them)\b", f"the {target_class}s", resolved, flags=re.IGNORECASE)
            return QuestionIntent.VISUAL_ATTRIBUTE, resolved, target_class, matched_group, None, explicit_threshold, is_follow_up

        # E. General Fallback
        target_class, matched_group = self.match_target_class_or_group(q_lower)
        return QuestionIntent.GENERAL, q, target_class, matched_group, None, explicit_threshold, is_follow_up


ContextManager = ConversationalContextManager
_context_manager_instance: Optional[ConversationalContextManager] = None


def get_context_manager() -> ConversationalContextManager:
    global _context_manager_instance
    if _context_manager_instance is None:
        _context_manager_instance = ConversationalContextManager()
    return _context_manager_instance
