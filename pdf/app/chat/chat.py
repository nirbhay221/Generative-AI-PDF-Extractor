from app.chat.models import ChatArgs
from app.chat.vector_stores import retriever_map
from app.chat.llms import llm_map
from app.chat.graph.graph import build_rag_graph
from app.chat.graph.chat import GraphChat
from app.chat.memories.histories.sql_history import SqlMessageHistory
from app.chat.score import random_component_by_score


def select_component(component_type, component_map, chat_args):
    # Lazy import to break app.chat ↔ app.web circular dependency
    from app.web.api import get_conversation_components
    components = get_conversation_components(chat_args.conversation_id)
    previous_component = components[component_type]

    if previous_component and previous_component in component_map:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args)

    random_name = random_component_by_score(component_type, component_map)
    builder = component_map[random_name]
    return random_name, builder(chat_args)


def build_chat(chat_args: ChatArgs):
    from app.web.api import set_conversation_components
    retriever_name, retriever = select_component("retriever", retriever_map, chat_args)
    llm_name, llm = select_component("llm", llm_map, chat_args)

    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever=retriever_name,
        memory="langgraph",
    )

    graph = build_rag_graph(llm, retriever)
    history = SqlMessageHistory(conversation_id=chat_args.conversation_id)
    return GraphChat(graph=graph, history=history)
