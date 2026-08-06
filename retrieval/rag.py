import os
import re
from typing import Annotated, TypedDict
from vectorstore.store import VectorStore
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from retrieval.prompts import SYSTEM_PROMPT, DIRECT_FILTER_PROMPT

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
)


def build_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = os.path.basename(meta["source"])
        if "start_time" in meta:
            mins, secs = int(meta["start_time"] // 60), int(meta["start_time"] % 60)
            citation = f"[Source {i}: {source} at {mins:02d}:{secs:02d}]"
        elif "page" in meta:
            citation = f"[Source {i}: {source} page {meta['page']}]"
        else:
            citation = f"[Source {i}: {source}]"
        parts.append(f"{citation}\n{r['text']}")
    return "\n\n".join(parts)


# ── tool factory — builds tools bound to a SPECIFIC store instance ─────────

def build_tools(store: VectorStore):
    """Create tool functions bound to this session's VectorStore."""

    @tool
    def semantic_search(query: str, source_type: str = "all") -> str:
        """Search documents by meaning. Use for general questions, topics, summaries, themes, morals, conclusions.
        Pass source_type as 'pdf', 'video', 'audio', 'docx', or 'all' to search everything (default)."""
        filter_val = None if source_type.lower() == "all" else source_type.lower()
        results = store.search(query, top_k=5, source_filter=filter_val)
        return build_context(results) if results else "No relevant information found."

    @tool
    def timestamp_search(query: str) -> str:
        """Find content at a specific time in video/audio. Pass the timestamp substring like '1:38'."""
        match = re.search(r'(\d+):(\d{2})', query)
        if not match:
            return "Could not parse a timestamp."
        seconds = int(match.group(1)) * 60 + int(match.group(2))
        all_chunks = store.collection.get(include=["documents", "metadatas"])
        results = []
        for i, meta in enumerate(all_chunks["metadatas"]):
            start = meta.get("start_time")
            end = meta.get("end_time")
            if start is None:
                continue
            chunk_end = end if end else start + 30.0
            if start <= seconds <= chunk_end:
                results.append({"text": all_chunks["documents"][i], "metadata": meta,
                                 "distance": 0.0, "id": all_chunks["ids"][i]})
        return build_context(results) if results else "No content found at that timestamp."

    @tool
    def page_search(query: str) -> str:
        """Find content from a specific page number. Pass the page number."""
        match = re.search(r'(\d+)', query)
        if not match:
            return "Could not parse a page number."
        page_number = int(match.group(1))
        all_chunks = store.collection.get(include=["documents", "metadatas"])
        results = []
        for i, meta in enumerate(all_chunks["metadatas"]):
            if meta.get("page") == page_number:
                results.append({"text": all_chunks["documents"][i], "metadata": meta,
                                 "distance": 0.0, "id": all_chunks["ids"][i]})
        return build_context(results) if results else f"No content found on page {page_number}."

    @tool
    def cross_source_search(query: str) -> str:
        """Search across ALL file types and combine results. Use for questions spanning multiple files or cross-references."""
        all_results, seen_ids = [], set()
        for source_type in ["pdf", "video", "audio", "docx"]:
            try:
                results = store.search(query, top_k=3, source_filter=source_type)
                for r in results:
                    if r["id"] not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r["id"])
            except Exception:
                continue
        all_results.sort(key=lambda x: x["distance"])
        return build_context(all_results[:8]) if all_results else "No content found across sources."

    @tool
    def list_sources(query: str = "") -> str:
        """List all currently indexed files."""
        try:
            all_data = store.collection.get(include=["metadatas"])
            sources = set(os.path.basename(m.get("source", "unknown")) for m in all_data["metadatas"])
            return "Indexed files: " + ", ".join(sources) if sources else "No files indexed yet."
        except Exception:
            return "Could not retrieve source list."

    return [semantic_search, timestamp_search, page_search, cross_source_search, list_sources]


# ── graph state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(store: VectorStore):
    """Build a LangGraph agent bound to this session's store."""
    tools = build_tools(store)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── public entry point ────────────────────────────────────────────────────

def ask(question: str, top_k: int = 4, source_filter: str = None, store: VectorStore = None) -> dict:
    if store is None:
        raise ValueError("A VectorStore instance must be provided to ask()")

    if source_filter and source_filter.lower() != "all":
        filter_val = source_filter.lower()
        results = store.search(question, top_k=top_k, source_filter=filter_val)
        if not results:
            return {"answer": f"No relevant information found in {source_filter} files.", "sources": []}
        context = build_context(results)
        response = llm.invoke([
            SystemMessage(content=DIRECT_FILTER_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ])
        return {"answer": response.content, "sources": []}

    try:
        app_graph = build_graph(store)
        result = app_graph.invoke({"messages": [HumanMessage(content=question)]})
        final_message = result["messages"][-1]
        return {"answer": final_message.content, "sources": []}
    except Exception as e:
        print(f"[Agent] Failed: {e}")
        results = store.search(question, top_k=5)
        if not results:
            return {"answer": "I couldn't find relevant information. Try uploading a file first.", "sources": []}
        return {"answer": build_context(results), "sources": []}