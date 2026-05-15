from langgraph.graph import StateGraph, END
from app.chat.graph.state import GraphState
from app.chat.graph.nodes import (
    make_condense_node,
    make_retrieve_node,
    make_grade_node,
    make_generate_node,
    make_rewrite_node,
)


def _decide_after_grade(state: GraphState) -> str:
    if state["documents"]:
        return "generate"
    elif state["retry_count"] < 2:
        return "rewrite"
    else:
        return "generate"


def build_rag_graph(llm, retriever):
    workflow = StateGraph(GraphState)

    workflow.add_node("condense", make_condense_node(llm))
    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("grade_documents", make_grade_node())
    workflow.add_node("generate", make_generate_node(llm))
    workflow.add_node("rewrite", make_rewrite_node())

    workflow.set_entry_point("condense")
    workflow.add_edge("condense", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        _decide_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()
