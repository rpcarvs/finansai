import praw
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from praw.reddit import Subreddit

from .utils import Classification

query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a social media expert and analyst. Call the tool once using the '_query_' \
            below exactly as the user provided. Do NOT modify it! \
            You HAVE TO call the reddit_tool once and ONLY once. \
            Extract social media messages from the Tool reply. \
            Analyze the messages as they are. You must be neutral. \
            --- \
            **IMPORTANT**: If no reply from the Tool, DO NOT INVENT any information. \
            Just write 'empty' for Summary. \
            --- \
            _query_ = {query} \
            Only extract Sentiment and perform a Summary, following the structure below and do \
            not add anything extra in your message: \
            -Query: Add here the query you used when calling the tool \
            -Sentiment: The sentiment of the messages in a scale from 0.0 (negative) to 5.0 (positive). \
            -Summary: A detailed summary representing the overall comments.",
        ),
        MessagesPlaceholder(variable_name="query"),
    ],
)


reddit = praw.Reddit(
    client_id=st.secrets.reddit_credentials.username,
    client_secret=st.secrets.reddit_credentials.password,
    user_agent="Comment Extraction/fincrw",
)

reddit.read_only = True


def get_posts_n_messages(
    query: str,
    subreddit: Subreddit,
    search_limit: int = 5,
    max_comments: int = 10,
) -> str:
    submission = None
    for submission in subreddit.search(
        query,
        sort="relevance",
        time_filter="month",
        limit=search_limit,
    ):
        # Limit, sort and Load comments
        submission.comment_sort = "top"
        submission.comment_limit = max_comments
        # Remove "load more comments" prompts
        submission.comments.replace_more(limit=0)

    if submission:
        serialized = "\n\n".join(
            (f"{comment.body}")  # type: ignore
            for comment in submission.comments.list()  # type: ignore
        )
        return serialized
    # if the search returns nothing
    return ""


@tool
def reddit_tool(query: str) -> str:
    """Use this tool to query messages from social media."""
    subs = ["StockMarket", "investing", "wallstreetbets"]

    serialized = ""
    all_comments = []
    for sub in subs:
        subreddit = reddit.subreddit(sub)
        comments = get_posts_n_messages(query, subreddit)
        all_comments.append(comments)

    serialized = "\n\n".join((f"{comment}") for comment in all_comments)
    return serialized


def query_social_agent(
    ticker: str,
    company: str,
    model: str,
) -> Classification:
    llm_social = ChatOllama(
        model=model,
        temperature=0.2,
    )
    agent_executor = create_react_agent(llm_social, [reddit_tool])

    structured_output = ChatOllama(
        model=model,
        temperature=0.2,
    ).with_structured_output(Classification)

    msg = query_prompt.invoke({"query": [f"{ticker} OR {company}"]})
    response = agent_executor.invoke(msg)
    return structured_output.invoke(
        response["messages"][-1].content.split("</think>")[-1],
    )  # type: ignore
