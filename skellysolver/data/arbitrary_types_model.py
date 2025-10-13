from pydantic import BaseModel, ConfigDict


class ArbitraryTypesModel(BaseModel):
    """Model with arbitrary types allowed."""

    model_config = ConfigDict(arbitrary_types_allowed= True)