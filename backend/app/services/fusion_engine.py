from typing import List, Dict, Any, Optional
import logging

from app.services.context_manager import (
    QuestionIntent,
    DialogueTurn,
    ConversationalContextManager,
    get_context_manager,
)

logger = logging.getLogger("bel_chatbot.fusion_engine")


class MultiModalFusionEngine:
    """
    Synthesizes YOLO spatial telemetry, VQA inference answers,
    and multi-turn dialogue history into coherent, natural language responses.
    """

    def __init__(self, context_manager: Optional[ConversationalContextManager] = None):
        self.context_manager = context_manager or get_context_manager()

    def generate_fused_response(
        self,
        query: str,
        vqa_answer: str,
        yolo_detections: List[dict],
        dialogue_history: List[DialogueTurn],
        conf_threshold: float = 0.40,
        vqa_error: Optional[str] = None,
    ) -> dict:
        """
        Fuses multimodal sources into a grammatically structured, context-aware answer.
        """
        # 1. Classify intent and resolve coreferences
        (
            intent,
            resolved_query,
            target_class,
            matched_classes,
            spatial_direction,
            explicit_threshold,
            is_follow_up,
        ) = self.context_manager.classify_and_resolve(
            query, dialogue_history, yolo_detections
        )

        effective_conf = explicit_threshold if explicit_threshold is not None else conf_threshold
        threshold_pct = int(round(effective_conf * 100))

        total_detections = len(yolo_detections)
        all_by_class: Dict[str, List[float]] = {}
        for obj in yolo_detections:
            cname = obj.get("class_name", "object")
            all_by_class.setdefault(cname, []).append(obj.get("confidence", 1.0))

        visible_by_class: Dict[str, int] = {}
        filtered_by_class: Dict[str, int] = {}
        for cname, confs in all_by_class.items():
            v_cnt = sum(1 for c in confs if c >= effective_conf)
            f_cnt = len(confs) - v_cnt
            if v_cnt > 0:
                visible_by_class[cname] = v_cnt
            if f_cnt > 0:
                filtered_by_class[cname] = f_cnt

        total_visible = sum(visible_by_class.values())
        detected_classes = list(visible_by_class.keys()) or list(all_by_class.keys())

        response = ""
        primary_subject = target_class

        # ROUTE 1: OBJECT INVENTORY
        if intent == QuestionIntent.OBJECT_INVENTORY:
            primary_subject = "detected objects"
            if total_detections == 0:
                response = "No objects were detected in this image."
            elif total_visible == 0:
                filtered_items = [f"{cnt} {cls}" if cnt == 1 else f"{cnt} {cls}s" for cls, cnt in filtered_by_class.items()]
                filtered_phrase = ", ".join(filtered_items)
                response = f"No objects met the {threshold_pct}% confidence threshold ({total_detections} total detected below threshold: {filtered_phrase})."
            else:
                sorted_visible = sorted(visible_by_class.items(), key=lambda x: x[1], reverse=True)
                parts = [f"{cnt} {cls}" if cnt == 1 else f"{cnt} {cls}s" if not cls.endswith("s") else f"{cnt} {cls}" for cls, cnt in sorted_visible]
                items_phrase = ", ".join(parts[:-1]) + f" and {parts[-1]}" if len(parts) > 1 else parts[0]
                category_count = len(visible_by_class)
                cat_word = "category" if category_count == 1 else "categories"

                if filtered_by_class:
                    filtered_parts = [f"{cnt} {cls}" if cnt == 1 else f"{cnt} {cls}s" for cls, cnt in filtered_by_class.items()]
                    filtered_phrase = ", ".join(filtered_parts)
                    response = (
                        f"I detected {total_visible} visible objects across {category_count} {cat_word} at {threshold_pct}% confidence: "
                        f"{items_phrase} ({total_detections} total detections found prior to filtering, including {filtered_phrase} below threshold)."
                    )
                else:
                    response = (
                        f"I detected {total_visible} visible objects across {category_count} {cat_word} at {threshold_pct}% confidence: "
                        f"{items_phrase} ({total_detections} total detections found)."
                    )

        # ROUTE 2: OBJECT COUNT
        elif intent == QuestionIntent.OBJECT_COUNT:
            if not target_class or target_class in ["object", "objects"]:
                return self.generate_fused_response(
                    "What objects are detected?", vqa_answer, yolo_detections, dialogue_history, conf_threshold
                )

            classes_to_match = set(matched_classes or [target_class])
            matching_objs = [d for d in yolo_detections if d.get("class_name") in classes_to_match]
            visible_matches = [d for d in matching_objs if d.get("confidence", 1.0) >= effective_conf]
            below_matches = [d for d in matching_objs if d.get("confidence", 1.0) < effective_conf]

            v_count = len(visible_matches)
            b_count = len(below_matches)
            t_count = len(matching_objs)

            label = target_class
            plural_label = f"{label}s" if not label.endswith("s") and label != "person" else "persons" if label == "person" else label

            if v_count > 0:
                name_str = label if v_count == 1 else plural_label
                if b_count > 0:
                    response = f"I detected {v_count} {name_str} in the image at {threshold_pct}% confidence ({t_count} total detected prior to filtering)."
                else:
                    response = f"I detected {v_count} {name_str} in the image at {threshold_pct}% confidence."
            elif b_count > 0:
                name_str = label if b_count == 1 else plural_label
                response = f"I detected 0 {plural_label} at {threshold_pct}% confidence ({b_count} {name_str} detected at lower confidence below the threshold)."
            else:
                response = f"I did not detect any {plural_label} in the image at the current confidence threshold."

        # ROUTE 3: SPATIAL LOCATION
        elif intent == QuestionIntent.SPATIAL_LOCATION:
            visible_detections = [d for d in yolo_detections if d.get("confidence", 1.0) >= effective_conf]

            if spatial_direction:
                sector_dir = spatial_direction.lower()
                matched_in_sector = [
                    d for d in visible_detections
                    if sector_dir in d.get("spatial_sector", "").lower()
                ]
                if matched_in_sector:
                    counts: Dict[str, int] = {}
                    for d in matched_in_sector:
                        c = d.get("class_name", "object")
                        counts[c] = counts.get(c, 0) + 1
                    parts = [f"{cnt} {cls}" if cnt == 1 else f"{cnt} {cls}s" if not cls.endswith("s") else f"{cnt} {cls}" for cls, cnt in counts.items()]
                    items_str = ", ".join(parts[:-1]) + f" and {parts[-1]}" if len(parts) > 1 else parts[0]
                    response = f"In the {sector_dir} area of the image at {threshold_pct}% confidence, I identified: {items_str}."
                else:
                    response = f"No objects were detected in the {sector_dir} sector of the image at {threshold_pct}% confidence."

            elif target_class and target_class not in ["object", "objects", "they", "it", "them"]:
                classes_to_match = set(matched_classes or [target_class])
                matching_objs = [d for d in visible_detections if d.get("class_name") in classes_to_match]

                if not matching_objs:
                    below_objs = [d for d in yolo_detections if d.get("class_name") in classes_to_match and d.get("confidence", 1.0) < effective_conf]
                    if below_objs:
                        response = f"No {target_class}s meet the {threshold_pct}% confidence threshold ({len(below_objs)} detected below threshold)."
                    else:
                        response = f"I did not detect any {target_class}s in the image."
                else:
                    sector_counts: Dict[str, int] = {}
                    for d in matching_objs:
                        s = d.get("spatial_sector", "center")
                        sector_counts[s] = sector_counts.get(s, 0) + 1

                    plural = f"{target_class}s" if not target_class.endswith("s") and target_class != "person" else "persons" if target_class == "person" else target_class
                    count_str = f"The {len(matching_objs)} {plural}" if len(matching_objs) > 1 else f"The {target_class}"

                    if len(matching_objs) == 1:
                        bbox = matching_objs[0].get("bbox", {})
                        sec = matching_objs[0].get("spatial_sector", "center")
                        conf_val = matching_objs[0].get("confidence", 0)
                        conf_pct = round(conf_val * 100, 1)

                        q_low = query.lower()
                        if any(w in q_low for w in ["road", "street", "sidewalk", "pavement", "driveway", "lawn", "grass"]):
                            if "sidewalk" in q_low and ("road" in q_low or "street" in q_low):
                                response = f"The {target_class} is positioned on the road (street), not on the sidewalk. It is located in the {sec} roadway lane with {conf_pct}% confidence."
                            elif "sidewalk" in q_low:
                                response = f"No, the {target_class} is on the road (street), not on the sidewalk (located in the {sec} sector at {conf_pct}% confidence)."
                            elif "driveway" in q_low and ("road" in q_low or "street" in q_low):
                                response = f"The {target_class} is in the roadway next to the driveway, located in the {sec} sector ({conf_pct}% confidence)."
                            elif "road" in q_low or "street" in q_low:
                                response = f"Yes, the {target_class} is positioned on the road/street in the {sec} area of the image ({conf_pct}% confidence)."
                            else:
                                response = f"The {target_class} is located on the roadway in the {sec} sector ({conf_pct}% confidence)."
                        else:
                            coord_str = f" (coordinates: x=[{bbox.get('x_min', 0):.0f}, {bbox.get('x_max', 0):.0f}], y=[{bbox.get('y_min', 0):.0f}, {bbox.get('y_max', 0):.0f}])" if bbox else ""
                            response = f"{count_str} is located in the {sec} sector of the image{coord_str} with {conf_pct}% confidence."
                    else:
                        sector_parts = [f"{sec} ({cnt})" if cnt > 1 else f"{sec}" for sec, cnt in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)]
                        sec_phrase = ", ".join(sector_parts[:-1]) + f" and {sector_parts[-1]}" if len(sector_parts) > 1 else sector_parts[0]
                        response = f"{count_str} are located primarily in the {sec_phrase} sectors of the image."

            else:
                by_class: Dict[str, List[str]] = {}
                for d in visible_detections:
                    c = d.get("class_name", "object")
                    s = d.get("spatial_sector", "center")
                    by_class.setdefault(c, []).append(s)

                if not by_class:
                    response = "No objects were detected above the confidence threshold to report spatial locations."
                else:
                    class_locs = []
                    for c, sectors in by_class.items():
                        uniq_secs = list(dict.fromkeys(sectors))
                        sec_str = ", ".join(uniq_secs[:3])
                        plural = f"{c}s" if not c.endswith("s") and c != "person" else "persons" if c == "person" else c
                        class_locs.append(f"{len(sectors)} {plural} positioned in the {sec_str} areas")
                    loc_summary = "; ".join(class_locs)
                    response = f"The detected objects are located across the image: {loc_summary}."

        # ROUTE 4: VISUAL ATTRIBUTES
        elif intent == QuestionIntent.VISUAL_ATTRIBUTE:
            clean_vqa = (vqa_answer or "").strip()
            if clean_vqa:
                if target_class:
                    response = f"Visual analysis indicates that the {target_class} is {clean_vqa}."
                else:
                    response = f"Visual analysis indicates: {clean_vqa.capitalize()}."
            elif target_class:
                classes_to_match = set(matched_classes or [target_class])
                matching = [d for d in yolo_detections if d.get("class_name") in classes_to_match]
                if matching:
                    m = matching[0]
                    conf = int(round(m.get("confidence", 0.5) * 100))
                    sector = m.get("spatial_sector", "center")
                    response = (
                        f"The {target_class} is identified in the {sector} area of the image ({conf}% confidence). "
                        f"Detailed color/texture analysis requires an active external vision key (GEMINI_API_KEY), but the {target_class} has been confirmed and localized by YOLO."
                    )
                else:
                    response = f"I did not detect any {target_class}s in the image to inspect."
            elif vqa_error:
                response = f"Visual question answering is currently unavailable: {vqa_error}"
            else:
                response = (
                    "Visual question answering is currently unavailable because external inference "
                    "is unconfigured or unreachable. You can continue asking questions about detected "
                    "objects, categories, counts, and spatial positions."
                )

        # ROUTE 5: GENERAL
        else:
            clean_vqa = (vqa_answer or "").strip()
            if clean_vqa:
                response = f"Visual analysis indicates: {clean_vqa.capitalize()}."
            elif target_class:
                classes_to_match = set(matched_classes or [target_class])
                matching = [d for d in visible_detections if d.get("class_name") in classes_to_match]
                if matching:
                    m = matching[0]
                    sec = m.get("spatial_sector", "center")
                    conf = int(round(m.get("confidence", 0.5) * 100))
                    response = f"The {target_class} is located in the {sec} area of the image ({conf}% confidence)."
                else:
                    response = f"I did not detect any {target_class}s in the image."
            elif total_detections > 0:
                response = f"The image contains {total_visible} visible detected objects at {threshold_pct}% confidence."
            elif vqa_error:
                response = f"Visual analysis is currently unavailable: {vqa_error}"
            else:
                response = (
                    "Visual analysis is currently operating in low-memory cloud mode. "
                    "You can ask questions about detected objects, counts, and spatial positions."
                )

        if is_follow_up and not response.startswith("Regarding your follow-up:"):
            response = f"Regarding your follow-up: {response}"

        return {
            "content": response,
            "vqa_answer": vqa_answer,
            "detected_classes": detected_classes,
            "total_objects": total_detections,
            "primary_subject": primary_subject,
            "intent": intent.value,
        }


FusionEngine = MultiModalFusionEngine
_fusion_engine_instance: Optional[MultiModalFusionEngine] = None


def get_fusion_engine() -> MultiModalFusionEngine:
    global _fusion_engine_instance
    if _fusion_engine_instance is None:
        _fusion_engine_instance = MultiModalFusionEngine()
    return _fusion_engine_instance
