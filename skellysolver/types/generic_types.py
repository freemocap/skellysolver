
TrackedPointNameString = str
KeypointNameString = str
SegmentName = str
DimensionName = str

TrackedPointList = list[TrackedPointNameString]
WeightsDict = dict[TrackedPointNameString|KeypointNameString, float]

KeypointMappingType = TrackedPointNameString| TrackedPointList| WeightsDict
# OffsetKeypoint = dict[Keypoint, Tuple[float, float, float]] # TODO - implement this

DimensionNames = list[DimensionName]

