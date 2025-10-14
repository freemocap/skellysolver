
KeypointNameString = str
SegmentName = str
DimensionName = str

WeightsDict = dict[KeypointNameString, float]

KeypointMappingType = KeypointNameString| list[KeypointNameString]| WeightsDict

DimensionNames = list[DimensionName]

