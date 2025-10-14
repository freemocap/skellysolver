from pydantic import BaseModel, ConfigDict


class ABaseModel(BaseModel):
    """Model with arbitrary types allowed."""
    model_config = ConfigDict(arbitrary_types_allowed= True)