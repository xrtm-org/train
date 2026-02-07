# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import logging

# From xrtm-forecast
from xrtm.forecast.kit.memory.unified import Memory as UnifiedMemory

logger = logging.getLogger(__name__)


class EpisodicLearner:
    def __init__(self, memory: UnifiedMemory):
        self.memory = memory

    def get_lessons_for_subject(self, subject_id: str, n_results: int = 3) -> str:
        try:
            query = f"Past performance and lessons for {subject_id}"
            experiences = self.memory.retrieve_similar(query, n_results=n_results)
            if not experiences:
                return "No relevant past experiences found."
            formatted_lessons = ["\n--- PAST LESSONS LEARNED ---\n"]
            for i, doc in enumerate(experiences, 1):
                formatted_lessons.append(f"Experience {i}:\n")
                formatted_lessons.append(doc.strip())
                formatted_lessons.append("\n")
            return "".join(formatted_lessons)
        except Exception as e:
            logger.error("[LEARNER] Failed to retrieve lessons: %s", e)
            return "Error retrieving past lessons."


__all__ = ["EpisodicLearner"]
