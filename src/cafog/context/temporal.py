from __future__ import annotations

from collections import deque

from .state import SceneState


class TemporalContextBuffer:
    """
    Maintains recent scene states.

    This is the first step toward temporal reasoning.

    Later this can feed a GRU/TCN/Transformer.
    """

    def __init__(self, max_length: int = 16) -> None:
        self.max_length = max_length
        self.buffer: deque[SceneState] = deque(
            maxlen=max_length
        )

    def add(self, state: SceneState) -> None:
        self.buffer.append(state)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    def latest(self) -> SceneState | None:
        if not self.buffer:
            return None

        return self.buffer[-1]

    def states(self) -> list[SceneState]:
        return list(self.buffer)

    def is_approaching_doorway(self) -> bool:
        """
        Initial placeholder for temporal reasoning.

        At this stage we simply check whether recent observations
        consistently contain a doorway.

        A learned temporal model can replace this later.
        """

        if len(self.buffer) < 2:
            return False

        return sum(
            state.door_detected
            for state in self.buffer
        ) >= 2