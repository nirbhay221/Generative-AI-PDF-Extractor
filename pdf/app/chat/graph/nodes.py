from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def make_condense_node(llm):
    condense_llm = ChatOpenAI(streaming=False, temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question that includes all necessary context.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow-up question: {question}\n\n"
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
        standalone = chain.invoke({
            "chat_history": history_text,
            "question": state["question"],
        })
        return {"question": standalone}

    return condense


def make_retrieve_node(retriever):
    def retrieve(state):
        docs = retriever.invoke(state["question"])
        return {"documents": docs}

    return retrieve


def make_grade_node():
    grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

    prompt = ChatPromptTemplate.from_template(
        "Is the following document relevant to answering the question? "
        "Answer only 'yes' or 'no'.\n\n"
        "Question: {question}\n\n"
        "Document: {document}"
    )

    chain = prompt | grader_llm | StrOutputParser()

    def grade_documents(state):
        question = state["question"]
        relevant = [
            doc for doc in state["documents"]
            if "yes" in chain.invoke({
                "question": question,
                "document": doc.page_content,
            }).lower()
        ]
        return {"documents": relevant}

    return grade_documents


def make_generate_node(llm):
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant answering questions based on provided document context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = prompt | llm | StrOutputParser()

    def generate(state):
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        answer = chain.invoke({
            "context": context,
            "question": state["question"],
        })
        return {"answer": answer}

    return generate


def make_rewrite_node():
    rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

    prompt = ChatPromptTemplate.from_template(
        "The previous search query did not return useful results. "
        "Rewrite the query to be more specific and likely to find relevant information.\n\n"
        "Original query: {question}\n\n"
        "Improved query:"
    )

    chain = prompt | rewrite_llm | StrOutputParser()

    def rewrite(state):
        new_question = chain.invoke({"question": state["question"]})
        return {"question": new_question, "retry_count": state["retry_count"] + 1}

    return rewrite
