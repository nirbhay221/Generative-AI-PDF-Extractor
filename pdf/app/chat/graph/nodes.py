import os
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.chat.graph.retry import invoke_with_retry


def make_query_router_node():
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

    prompt = ChatPromptTemplate.from_template(
        "You are a query router for a research-paper and PDF Q&A system.\n\n"
        "Route the question to exactly one of:\n"
        "  - 'retrieval'  : question is about the uploaded document's specific content, "
        "methods, results, authors, citations, or technical details\n"
        "  - 'direct'     : trivial general-knowledge question (e.g. basic math, "
        "common definitions not needing the paper)\n"
        "  - 'web_search' : requires current/real-time data that cannot be in any static document "
        "(e.g. latest news, stock prices, live datasets)\n\n"
        "When in doubt, choose 'retrieval'.\n\n"
        "Question: {question}\n\n"
        "Route (output only the word):"
    )

    chain = prompt | router_llm | StrOutputParser()

    def route_query(state):
        result = invoke_with_retry(chain, {"question": state["question"]}).lower().strip()
        if "web" in result:
            route = "web_search"
        elif "direct" in result:
            route = "direct"
        else:
            route = "retrieval"
        logging.info(f"[router] '{state['question'][:60]}...' -> {route}")
        return {"route": route}

    return route_query


def make_condense_node(llm):
    condense_llm = ChatOpenAI(streaming=False, temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Given the conversation history and a follow-up question, rephrase the "
        "follow-up as a fully self-contained standalone question.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow-up: {question}\n\n"
        "Standalone question:"
    )

    chain = prompt | condense_llm | StrOutputParser()

    def condense(state):
        if not state["chat_history"]:
            return {"question": state["question"]}

        history_text = "\n".join(
            f"{'Human' if msg.type == 'human' else 'Assistant'}: {msg.content}"
            for msg in state["chat_history"]
        )
        standalone = invoke_with_retry(chain, {
            "chat_history": history_text,
            "question": state["question"],
        })
        return {"question": standalone}

    return condense


def make_retrieve_node(retriever):
    def retrieve(state):
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "source": "pdf"}

    return retrieve


def make_grade_node():
    grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

    prompt = ChatPromptTemplate.from_template(
        "You are a relevance grader for a research-paper RAG system.\n\n"
        "Does this document chunk contain information that is useful - even partially - "
        "for answering the question?\n"
        "Be lenient: answer 'yes' if the chunk provides any relevant facts, "
        "definitions, context, or reasoning.\n\n"
        "Question: {question}\n\n"
        "Chunk: {document}\n\n"
        "Answer (yes/no):"
    )

    chain = prompt | grader_llm | StrOutputParser()

    def grade_documents(state):
        question = state["question"]

        def is_relevant(doc):
            result = invoke_with_retry(chain, {
                "question": question,
                "document": doc.page_content,
            })
            return "yes" in result.lower()

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(is_relevant, state["documents"]))

        relevant = [doc for doc, ok in zip(state["documents"], results) if ok]
        logging.info(f"[grader] {len(relevant)}/{len(state['documents'])} chunks relevant")
        return {"documents": relevant}

    return grade_documents


def make_rewrite_node():
    rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

    prompt = ChatPromptTemplate.from_template(
        "The previous query returned no useful results from a research paper or technical PDF.\n"
        "Rewrite the query to be more specific, using different terminology or phrasing "
        "that is more likely to match content in an academic document.\n\n"
        "Original query: {question}\n\n"
        "Improved query:"
    )

    chain = prompt | rewrite_llm | StrOutputParser()

    def rewrite(state):
        new_question = invoke_with_retry(chain, {"question": state["question"]})
        logging.info(f"[rewrite] retry_count={state['retry_count']+1}")
        return {"question": new_question, "retry_count": state["retry_count"] + 1}

    return rewrite


def make_web_search_node():
    search = None
    try:
        from langchain_tavily import TavilySearch
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            search = TavilySearch(max_results=4, search_depth="advanced")
        else:
            logging.warning("[web_search] TAVILY_API_KEY not set - web search disabled")
    except ImportError:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tavily_key = os.environ.get("TAVILY_API_KEY")
            if tavily_key:
                search = TavilySearchResults(max_results=4, search_depth="advanced")
        except ImportError:
            logging.warning("[web_search] langchain-tavily not installed - web search disabled")

    def web_search(state):
        if not search:
            logging.warning("[web_search] TAVILY_API_KEY not set - skipping web search")
            return {"documents": [], "source": "web"}
        try:
            results = search.invoke(state["question"])
            web_docs = [
                Document(
                    page_content=r["content"],
                    metadata={"source": r["url"], "type": "web"},
                )
                for r in results
            ]
            logging.info(f"[web_search] retrieved {len(web_docs)} results")
        except Exception as e:
            logging.error(f"[web_search] failed: {e}")
            web_docs = []
        return {"documents": web_docs, "source": "web"}

    return web_search


def make_generate_node(llm):
    prompt = ChatPromptTemplate.from_template(
        "You are a precise assistant answering questions from {source}.\n\n"
        "Context (each chunk labelled with its origin):\n{context}\n\n"
        "Question: {question}\n\n"
        "Instructions:\n"
        "- For PDF content, cite page numbers inline (e.g. 'As stated on page 3...')\n"
        "- For web results, mention the source domain when relevant\n"
        "- If the context is insufficient, clearly say so rather than guessing\n"
        "- Be precise and thorough for research/technical questions\n\n"
        "Answer:"
    )

    chain = prompt | llm | StrOutputParser()

    def _format_context(docs):
        parts = []
        for doc in docs:
            page = doc.metadata.get("page")
            url = doc.metadata.get("source", "")
            if url and url.startswith("http"):
                parts.append(f"[Source: {url}]\n{doc.page_content}")
            elif page is not None:
                parts.append(f"[Page {int(page) + 1}]\n{doc.page_content}")
            else:
                parts.append(doc.page_content)
        return "\n\n".join(parts)

    def generate(state):
        source_label = (
            "the uploaded PDF document"
            if state.get("source") == "pdf"
            else "web search results"
        )
        answer = invoke_with_retry(chain, {
            "context": _format_context(state["documents"]),
            "question": state["question"],
            "source": source_label,
        })
        return {"answer": answer}

    return generate


def make_direct_generate_node(llm):
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question using your knowledge.\n"
        "Be concise and accurate.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = prompt | llm | StrOutputParser()

    def generate_direct(state):
        answer = invoke_with_retry(chain, {"question": state["question"]})
        return {"answer": answer, "source": "llm_direct"}

    return generate_direct


def make_hallucination_grader_node():
    grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

    prompt = ChatPromptTemplate.from_template(
        "You are a hallucination detector for a research-paper RAG system.\n\n"
        "Determine if the answer is grounded in the provided source documents.\n"
        "Answer 'yes' if every significant claim in the answer is supported by the documents.\n"
        "Answer 'no' if the answer contains facts, numbers, or claims not present in the documents.\n"
        "Do not penalise for reasonable inferences that follow from the documents.\n\n"
        "Source documents:\n{documents}\n\n"
        "Answer to check:\n{answer}\n\n"
        "Is the answer grounded in the documents? (yes/no):"
    )

    chain = prompt | grader_llm | StrOutputParser()

    def grade_hallucination(state):
        if not state["documents"]:
            logging.info("[hallucination_grader] no documents - skipping grounding check")
            return {"hallucination_grade": "yes"}

        docs_text = "\n\n".join(doc.page_content for doc in state["documents"])
        result = invoke_with_retry(chain, {
            "documents": docs_text,
            "answer": state["answer"],
        })
        grounded = "yes" if "yes" in result.lower() else "no"
        updates = {"hallucination_grade": grounded}
        if grounded == "no":
            new_count = state.get("generate_retry_count", 0) + 1
            updates["generate_retry_count"] = new_count
            logging.warning(f"[hallucination_grader] not grounded - retry {new_count}/3")
        else:
            logging.info("[hallucination_grader] answer is grounded")
        return updates

    return grade_hallucination


def make_answer_grader_node():
    grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

    prompt = ChatPromptTemplate.from_template(
        "Does the following answer directly and completely address the question?\n\n"
        "Question: {question}\n\n"
        "Answer: {answer}\n\n"
        "Answer 'yes' if the response resolves the question, 'no' if it is "
        "off-topic, incomplete, or evasive.\n\n"
        "Useful? (yes/no):"
    )

    chain = prompt | grader_llm | StrOutputParser()

    def grade_answer(state):
        result = invoke_with_retry(chain, {
            "question": state["original_question"],
            "answer": state["answer"],
        })
        useful = "yes" if "yes" in result.lower() else "no"
        updates = {"answer_grade": useful}
        if useful == "no":
            new_count = state.get("answer_retry_count", 0) + 1
            updates["answer_retry_count"] = new_count
            logging.warning(f"[answer_grader] answer not useful - rewrite attempt {new_count}/1")
        else:
            logging.info("[answer_grader] answer is useful")
        return updates

    return grade_answer
