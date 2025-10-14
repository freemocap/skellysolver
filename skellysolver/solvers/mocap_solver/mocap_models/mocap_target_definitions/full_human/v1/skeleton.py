

class BodySkeletonDefinition(SkeletonABC):
    parent = BodyChains.AXIAL
    children = [BodyChains.RIGHT_ARM,
                BodyChains.RIGHT_LEG,
                BodyChains.LEFT_ARM,
                BodyChains.LEFT_LEG,
                ]
