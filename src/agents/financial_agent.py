import os
from typing import Literal

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_tavily.tavily_search import TavilySearch
from langgraph.prebuilt import create_react_agent

from .utils import Classification

os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial news and stock market expert. Call the tool once using the '_query_' \
            below exactly as the user provided. Do NOT modify it! \
            You HAVE TO call the tav_search tool once and ONLY once. \
            _query_ = {query} \
            Extract information from the Tool reply. \
            --- \
            **IMPORTANT**: If no reply from the Tool, DO NOT INVENT any information. \
            Just write 'empty' for Summary. \
            --- \
            Only extract Sentiment and perform a Summary, following the structure below and do \
            not add anything extra in your message: \
            -Query: Add here the query you used when calling the tool \
            -Sentiment: The sentiment of the messages in a scale from 0.0 (negative) to 5.0 (positive). \
            -Summary: A detailed and long summary representing the retrieved information.",
        ),
        MessagesPlaceholder(variable_name="query"),
    ]
)

tav_search = TavilySearch(
    max_results=5,
    days=30,
    topic="news",
    include_answer="advanced",
    search_depth="advanced",
)


def query_financial_agent(
    ticker: str,
    company: str,
    model: Literal["qwen3:1.7b", "qwen3:0.6b", "llama3.2:3b"] = "llama3.2:3b",
) -> Classification:
    llm_financial = ChatOllama(
        model=model,
        temperature=0.2,
    )
    agent_executor = create_react_agent(llm_financial, [tav_search])

    structured_output = ChatOllama(
        model=model,
        temperature=0.2,
    ).with_structured_output(Classification)

    msg = query_prompt.invoke(
        {
            "query": [
                f"What are the most relevant financial news about {ticker} or {company}?"
            ]
        }
    )
    response = agent_executor.invoke(msg)
    return structured_output.invoke(
        response["messages"][-1].content.split("</think>")[-1]
    )  # type: ignore
