from pydantic import BaseModel, Field


class Classification(BaseModel):
    query: str = Field(description="Add here exatcly the text from the 'query' field")
    sentiment: float = Field(
        description="Add here exatcly what is written in the Sentiment part of the message"
    )
    summary: str = Field(
        description="Add here exatcly what is written in the Summary part of the message"
    )
