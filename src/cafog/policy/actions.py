
from enum import StrEnum


class ActionType(StrEnum):
    """Actions the system may recommend."""

    CONTINUE = "CONTINUE"
    CENTER_PATH = "CENTER_PATH"
    FOLLOW_SAFE_PATH = "FOLLOW_SAFE_PATH"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    PAUSE_AND_ORIENT = "PAUSE_AND_ORIENT"
    NO_AUTOMATIC_ACTION = "NO_AUTOMATIC_ACTION"
