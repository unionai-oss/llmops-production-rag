"""Agentic RAG implementation of the Union and Flyte chat assistant."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from flytekit import dynamic, task, workflow, Secret
from flytekit.deck import DeckField
from flytekit.deck.renderer import MarkdownRenderer
from flytekit.types.directory import FlyteDirectory
from union.actor import ActorEnvironment

from llmops_rag.simple_rag import image, VectorStore
from llmops_rag.utils import openai_env_secret


actor = ActorEnvironment(
    name="agentic-rag",
    ttl_seconds=720,
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)

# maximum number of question rewrites
MAX_REWRITES = 10


class AgentAction(Enum):
    tools = "tools"
    end = "end"


class GraderAction(Enum):
    generate = "generate"
    rewrite = "rewrite"


@dataclass
class Message:
    """Json-encoded message."""

    data: str

    def to_langchain(self):
        from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

        data = json.loads(self.data)
        message_type = data.get("type", data.get("role"))
        return {
            "ai": AIMessage,
            "tool": ToolMessage,
            "human": HumanMessage,
        }[message_type](**data)

    @classmethod
    def from_langchain(cls, message):
        return cls(data=json.dumps(message.dict()))


@dataclass
class AgentState:
    """A list of messages capturing the state of the RAG execution graph."""

    messages: list[Message]

    def to_langchain(self) -> dict:
        return {"messages": [message.to_langchain() for message in self.messages]}

    def append(self, message):
        self.messages.append(Message.from_langchain(message))

    def __getitem__(self, index):
        message: Message = self.messages[index]
        return message.to_langchain()


def get_vector_store_retriever(path: str):
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain.tools.retriever import create_retriever_tool

    retriever = FAISS.load_local(
        path,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    ).as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_flytekit_docs",
        "Search and return information about flytekit docs to the user query.",
    )
    return retriever_tool


@actor.task
@openai_env_secret
def retriever_agent(
    state: AgentState,
    search_index: FlyteDirectory,
) -> tuple[AgentState, AgentAction]:
    """Invokes the agent to either end the loop or call the retrieval tool."""

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    search_index.download()
    retriever_tool = get_vector_store_retriever(search_index.path)

    prompt = PromptTemplate(
        template="""You are a helpful chat assistant that is an expert in Flyte
        and the flytekit sdk.

        Here is the user question: {question} \n

        If the question is related to flyte or flytekit, call the relevant
        tool that you have access to. If the question is not related to
        flyte or flytekit, end the loop with a response that the question
        is not relevant.""",
        input_variables=["question"],
    )

    question_message = state[-1]
    assert question_message.type == "human"

    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools([retriever_tool])
    chain = prompt | model
    response = chain.invoke({"question": question_message.content})

    # Get agent's decision to call the retrieval tool or end the loop
    action = AgentAction.end
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        action = AgentAction.tools

    state.append(response)
    return state, action


@actor.task
@openai_env_secret
def retrieve(
    state: AgentState,
    search_index: FlyteDirectory,
) -> AgentState:
    """Retrieves documents from the vector store."""

    from langchain_core.messages import AIMessage, ToolMessage

    search_index.download()
    retriever_tool = get_vector_store_retriever(search_index.path)

    agent_message = state[-1]
    assert isinstance(agent_message, AIMessage)
    assert len(agent_message.tool_calls) == 1

    # invoke the tool to retrieve documents from the vector store
    tool_call = agent_message.tool_calls[0]
    content = retriever_tool.invoke(tool_call["args"])
    response = ToolMessage(content=content, tool_call_id=tool_call["id"])
    state.append(response)
    return state


@actor.task
@openai_env_secret
def grader_agent(state: AgentState) -> GraderAction:
    """Determines whether the retrieved documents are relevant to the question."""

    from langchain_core.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    # Restrict the LLM's output to be a binary "yes" or "no"
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved
        document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the
        user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the
        document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state.to_langchain()["messages"]

    # get the last "human" and "tool" message, which contains the question and
    # retrieval tool context, respectively
    questions = [m for m in messages if m.type == "human"]
    contexts = [m for m in messages if m.type == "tool"]
    question = questions[-1]
    context = contexts[-1]

    scored_result = chain.invoke(
        {
            "question": question.content,
            "context": context.content,
        }
    )
    score = scored_result.binary_score
    return {
        "yes": GraderAction.generate,
        "no": GraderAction.rewrite,
    }[score]


@actor.task
@openai_env_secret
def rewrite(state: AgentState) -> AgentState:
    """Transform the query to produce a better question."""

    from langchain_core.messages import HumanMessage
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    messages = state.to_langchain()["messages"]

    # get the last "human", which contains the user question
    questions = [m for m in messages if m.type == "human"]
    question = questions[-1].content

    class rewritten_question(BaseModel):
        """Binary score for relevance check."""

        question: str = Field(description="Rewritten question")
        reason: str = Field(description="Reasoning for the rewrite")

    rewrite_prompt = f"""
    Look at the input and try to reason about the underlying semantic
    intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question and provide your reasoning.
    """

    # define model with structured output for the question rewrite
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    rewriter_model = model.with_structured_output(rewritten_question)

    response = rewriter_model.invoke([HumanMessage(content=rewrite_prompt)])
    message = HumanMessage(
        content=response.question,
        response_metadata={"rewrite_reason": response.reason},
    )
    state.append(message)
    return state


@actor.task
@openai_env_secret
def generate(state: AgentState) -> AgentState:
    """Generate an answer based on the state."""

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    messages = state.to_langchain()["messages"]

    # get the last "human" and "tool" message, which contains the question and
    # retrieval tool context, respectively
    questions = [m for m in messages if m.type == "human"]
    contexts = [m for m in messages if m.type == "tool"]
    question = questions[-1]
    context = contexts[-1]

    system_message = """
    You are an assistant for question-answering tasks about flyte and flytekit.
    Use the following pieces of retrieved context to answer the question. If you
    don't know the answer, just say that you don't know. Make the answer as
    detailed as possible. If the answer contains acronyms or jargon, make sure
    to expand on them.

    Question: {question}

    Context: {context}

    Answer:
    """

    prompt = ChatPromptTemplate.from_messages([("human", system_message)])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke(
        {
            "context": context.content,
            "question": question.content,
        }
    )
    if isinstance(response, str):
        response = AIMessage(response)

    state.append(response)
    return state


@task(container_image=image, enable_deck=True, deck_fields=[DeckField.OUTPUT])
def return_answer(state: AgentState) -> Annotated[str, MarkdownRenderer()]:
    """Finalize the answer to return a string to the user."""

    if len(state.messages) == 1:
        return f"I'm sorry, I don't understand: '{state.messages}'"
    else:
        data = state.messages[-1].to_langchain()
        return data.content


@dynamic(container_image=image)
def retrieval_router(
    state: AgentState,
    action: AgentAction,
    search_index: FlyteDirectory,
    n_rewrites: int,
) -> AgentState:
    """
    The first conditional branch in the RAG workflow. This determines whether
    the agent loop should end or call the retrieval tool for grading.
    """

    if action == AgentAction.end:
        return state
    elif action == AgentAction.tools:
        state = retrieve(state, search_index)
        grader_action = grader_agent(state)
        return grading_router(state, grader_action, search_index, n_rewrites)
    else:
        raise RuntimeError(f"Invalid action '{action}'")


@dynamic(container_image=image)
def grading_router(
    state: AgentState,
    grader_action: GraderAction,
    search_index: FlyteDirectory,
    n_rewrites: int,
) -> AgentState:
    """
    The second conditional branch in the RAG workflow. This determines whether
    the rewrite the original user's query or generate the final answer.
    """
    if grader_action == GraderAction.generate or n_rewrites >= MAX_REWRITES:
        return generate(state)
    elif grader_action == GraderAction.rewrite:
        state = rewrite(state)
        state, action = retriever_agent(state, search_index)
        return retrieval_router(state, action, search_index, n_rewrites + 1)
    else:
        raise RuntimeError(f"Invalid action '{grader_action}'")


@actor.task(cache=True, cache_version="0")
def init_state(question: str) -> AgentState:
    """Initialize the AgentState with the user's message."""
    from langchain_core.messages import HumanMessage

    return AgentState(messages=[Message.from_langchain(HumanMessage(question))])


@workflow
def run(
    question: str,
    vector_store: FlyteDirectory = VectorStore.query(),
) -> str:
    """An agentic retrieval augmented generation workflow."""
    state = init_state(user_message=question)
    state, action = retriever_agent(state=state, vector_store=vector_store)
    state = retrieval_router(state, action, vector_store, n_rewrites=0)
    return return_answer(state=state)
