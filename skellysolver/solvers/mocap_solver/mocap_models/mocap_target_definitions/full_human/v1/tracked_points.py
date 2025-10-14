from enum import auto

from skelly_blender.core.pure_python.utility_classes.lowercaseable_enum import LowercaseableEnum


class MediapipeBodyPoints(LowercaseableEnum):
    NOSE = auto()
    LEFT_EYE_INNER = auto()
    LEFT_EYE = auto()
    LEFT_EYE_OUTER = auto()
    RIGHT_EYE_INNER = auto()
    RIGHT_EYE = auto()
    RIGHT_EYE_OUTER = auto()
    LEFT_EAR = auto()
    RIGHT_EAR = auto()
    MOUTH_LEFT = auto()
    MOUTH_RIGHT = auto()
    LEFT_SHOULDER = auto()
    RIGHT_SHOULDER = auto()
    LEFT_ELBOW = auto()
    RIGHT_ELBOW = auto()
    LEFT_WRIST = auto()
    RIGHT_WRIST = auto()
    LEFT_PINKY = auto()
    RIGHT_PINKY = auto()
    LEFT_INDEX = auto()
    RIGHT_INDEX = auto()
    LEFT_THUMB = auto()
    RIGHT_THUMB = auto()
    LEFT_HIP = auto()
    RIGHT_HIP = auto()
    LEFT_KNEE = auto()
    RIGHT_KNEE = auto()
    LEFT_ANKLE = auto()
    RIGHT_ANKLE = auto()
    LEFT_HEEL = auto()
    RIGHT_HEEL = auto()
    LEFT_FOOT_INDEX = auto()
    RIGHT_FOOT_INDEX = auto()


class MediapipeHandPoints(LowercaseableEnum):
    WRIST = auto()
    THUMB_CMC = auto()
    THUMB_MCP = auto()
    THUMB_IP = auto()
    THUMB_TIP = auto()

    INDEX_FINGER_MCP = auto()
    INDEX_FINGER_PIP = auto()
    INDEX_FINGER_DIP = auto()
    INDEX_FINGER_TIP = auto()

    MIDDLE_FINGER_MCP = auto()
    MIDDLE_FINGER_PIP = auto()
    MIDDLE_FINGER_DIP = auto()
    MIDDLE_FINGER_TIP = auto()

    RING_FINGER_MCP = auto()
    RING_FINGER_PIP = auto()
    RING_FINGER_DIP = auto()
    RING_FINGER_TIP = auto()

    PINKY_MCP = auto()
    PINKY_PIP = auto()
    PINKY_DIP = auto()
    PINKY_TIP = auto()

