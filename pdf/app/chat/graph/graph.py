import logging
from langgraph.graph import StateGraph, END

from app.chat.graph.state import GraphState
from app.chat.graph.nodes import (
    make_condense_node,
    make_query_router_node,
    make_retrieve_node,
    make_grade_node,
    make_rewrite_node,
    make_web_search_node,
    make_generate_node,
    make_direct_generate_node,
    make_hallucination_grader_node,
    make_answer_grader_node,
)


def _route_from_router(state: GraphState) -> str:
    return state["route"]


def _decide_after_grade(state: GraphState) -> str:
    if state["documents"]:
        return "generate"
    elif state["retry_count"] < 1:
        return "rewrite"
    else:
        logging.info("[graph] retrieval exhausted - falling back to web search")
        return "web_search"


def _decide_after_hallucination(state: GraphState) -> str:
    if state["hallucination_grade"] == "yes":
        return "answer_grader"
    elif state.get("generate_retry_count", 0) <= 2:
        return "generate"
    else:
        logging.warning("[graph] max generation retries reached - returning best-effort answer")
        return "end"


def _decide_after_answer(state: GraphState) -> str:
    if state["answer_grade"] == "yes":
        return "end"
    elif state.get("answer_retry_count", 0) <= 1:
        return "rewrite"
    else:
        logging.warning("[graph] answer quality retry exhausted - returning current answer")
        return "end"


def build_rag_graph(llm, retriever):
    workflow = StateGraph(GraphState)

    workflow.add_node("condense", make_condense_node(llm))
    workflow.add_node("query_router", make_query_router_node())
    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("grade_documents", make_grade_node())
    workflow.add_node("rewrite", make_rewrite_node())
    workflow.add_node("web_search", make_web_search_node())
    workflow.add_node("generate", make_generate_node(llm))
    workflow.add_node("generate_direct", make_direct_generate_node(llm))
    workflow.add_node("hallucination_grader", make_hallucination_grader_node())
    workflow.add_node("answer_grader", make_answer_grader_node())

    workflow.set_entry_point("condense")

    workflow.add_edge("condense", "query_router")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "hallucination_grader")
    workflow.add_edge("generate_direct", END)

    workflow.add_conditional_edges(
        "query_router",
        _route_from_router,
        {"retrieval": "retrieve", "direct": "generate_direct", "web_search": "web_search"},
    )

    workflow.add_conditional_edges(
        "grade_documents",
        _decide_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )

    workflow.add_conditional_edges(
        "hallucination_grader",
        _decide_after_hallucination,
        {"answer_grader": "answer_grader", "generate": "generate", "end": END},
    )

    workflow.add_conditional_edges(
        "answer_grader",
        _decide_after_answer,
        {"end": END, "rewrite": "rewrite"},
    )

    return workflow.compile()
