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

from retrieval.prompts import SYSTEM_PROMPT, DIRECT_FILTER_PROMPT, FALLBACK_PROMPT

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b"),
    temperature=0.2
)


def fmt_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


class SourceTracker:
    """Numbers chunks and remembers them, so the UI can show what was retrieved."""

    def __init__(self):
        self.sources = []
        self._numbers = {}  # chunk id -> source number

    def context(self, results: list) -> str:
        parts = []
        for r in results:
            meta = r["metadata"]
            cid = r["id"]
            name = os.path.basename(meta["source"])

            if cid not in self._numbers:
                n = len(self.sources) + 1
                self._numbers[cid] = n
                info = {"n": n, "file": name, "type": meta["source_type"]}
                if "start_time" in meta:
                    info["timestamp"] = fmt_time(meta["start_time"])
                if "page" in meta:
                    info["page"] = meta["page"]
                if r["distance"] > 0:  # exact lookups have no similarity score
                    info["relevance"] = round(1 - r["distance"], 2)
                self.sources.append(info)

            n = self._numbers[cid]
            if "start_time" in meta:
                label = f"[Source {n}: {name} at {fmt_time(meta['start_time'])}]"
            elif "page" in meta:
                label = f"[Source {n}: {name} page {meta['page']}]"
            else:
                label = f"[Source {n}: {name}]"
            parts.append(f"{label}\n{r['text']}")
        return "\n\n".join(parts)


def build_tools(store: VectorStore, tracker: SourceTracker):
    """Tools bound to this session's store and this question's tracker."""

    @tool
    def semantic_search(query: str, source_type: str = "all") -> str:
        """Search documents by meaning. Use for general questions, topics, summaries, themes, morals, conclusions.
        Pass source_type as 'pdf', 'video', 'audio', 'docx', or 'all' to search everything (default)."""
        filter_val = None if source_type.lower() == "all" else source_type.lower()
        results = store.search(query, top_k=5, source_filter=filter_val)
        return tracker.context(results) if results else "No relevant information found."

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
            if start is None:
                continue
            end = meta.get("end_time") or start + 30.0
            if start <= seconds <= end:
                results.append({"text": all_chunks["documents"][i], "metadata": meta,
                                "distance": 0.0, "id": all_chunks["ids"][i]})
        return tracker.context(results) if results else "No content found at that timestamp."

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
        return tracker.context(results) if results else f"No content found on page {page_number}."

    @tool
    def cross_source_search(query: str) -> str:
        """Search across ALL file types and combine results. Use for questions spanning multiple files or cross-references."""
        all_results, seen_ids = [], set()
        for source_type in ["pdf", "video", "audio", "docx"]:
            try:
                for r in store.search(query, top_k=3, source_filter=source_type):
                    if r["id"] not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r["id"])
            except Exception:
                continue
        all_results.sort(key=lambda x: x["distance"])
        return tracker.context(all_results[:8]) if all_results else "No content found across sources."

    @tool
    def list_sources(query: str = "") -> str:
        """List all currently indexed files."""
        try:
            all_data = store.collection.get(include=["metadatas"])
            names = set(os.path.basename(m.get("source", "unknown")) for m in all_data["metadatas"])
            return "Indexed files: " + ", ".join(names) if names else "No files indexed yet."
        except Exception:
            return "Could not retrieve source list."

    return [semantic_search, timestamp_search, page_search, cross_source_search, list_sources]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(store: VectorStore, tracker: SourceTracker):
    tools = build_tools(store, tracker)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        try:
            data = store.collection.get(include=["metadatas"])
            names = sorted({os.path.basename(m["source"]) for m in data["metadatas"]})
        except Exception:
            names = []
        status = (
            f"Indexed files right now: {', '.join(names)}. Use a search tool to answer questions about them."
            if names else "No files are indexed yet."
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT + "\n\n" + status)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def ask(question: str, top_k: int = 4, source_filter: str = None, store: VectorStore = None) -> dict:
    if store is None:
        raise ValueError("A VectorStore instance must be provided to ask()")

    tracker = SourceTracker()

    # explicit filter from the UI: skip the agent, search directly
    if source_filter and source_filter.lower() != "all":
        results = store.search(question, top_k=top_k, source_filter=source_filter.lower())
        if not results:
            return {"answer": f"No relevant information found in {source_filter} files.", "sources": []}
        context = tracker.context(results)
        response = llm.invoke([
            SystemMessage(content=FALLBACK_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nMessage: {question}")
        ])
        return {"answer": response.content, "sources": tracker.sources}

    try:
        app_graph = build_graph(store, tracker)
        result = app_graph.invoke({"messages": [HumanMessage(content=question)]})
        print("[Agent] tool calls:", [tc["name"] for m in result["messages"] for tc in (getattr(m, "tool_calls", None) or [])])
        return {"answer": result["messages"][-1].content, "sources": tracker.sources}
    except Exception as e:
        print(f"[Agent] Failed: {e}")
        results = store.search(question, top_k=5)
        context = tracker.context(results) if results else "No files indexed."
        response = llm.invoke([
            SystemMessage(content=(
                "You are AnyChat AI, a document assistant. If the message is just a "
                "greeting or small talk, reply briefly and ignore the context. "
                "Otherwise answer from the context and cite [Source N]."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nMessage: {question}")
        ])
        answer = response.content
        # only show sources if the answer actually used them
        return {"answer": answer, "sources": tracker.sources if "[Source" in answer else []}