from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from presidio_analyzer import Pattern, PatternRecognizer
import os
from langchain_groq import ChatGroq
from typing import Any, List


def user_login() -> int:
    """Login function"""
    while True:
        try:
            return int(input("Enter your ID: "))
        except ValueError:
            print("Invalid input. Please enter a valid ID.")


def pesel_recognition(analyzer: AnalyzerEngine) -> None:
    """Additional filter to recognize PESEL"""
    pesel_pattern: Pattern = Pattern(name="pesel_regex", regex=r"\b\d{11}\b", score=0.9)

    pesel_recognizer: PatternRecognizer = PatternRecognizer(supported_entity="ID_NUM", patterns=[pesel_pattern])

    analyzer.registry.add_recognizer(pesel_recognizer)


def pseudoanonymize(analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine, vul_text: str) -> Any:
    """Transforming vulnerable data to safe"""
    vul_data: List[Any] = analyzer.analyze(
        text=vul_text,
        entities=["PERSON", "ID_NUM"],
        language="en"
    )

    safe_data: Any = anonymizer.anonymize(
        text=vul_text,
        analyzer_results=vul_data
    )

    return safe_data


def user_question(active_id: int, vector_base: Chroma, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine, llm: ChatGroq) -> None:
    """Prompt look and conversation history"""
    history: List[str] = []
    max_questions: int = 3
    while True:
        question: str = input("Ask what you want! (or type stop to exit): ")

        if question.lower() == 'stop':
            print("Bye!")
            break

        results: List[Any] = vector_base.similarity_search(query=question, filter={"user_id": active_id})

        if not results:
            print("User not found in the database")
            continue

        extracted_text: str = results[0].page_content
        extracted_department: str = results[0].metadata["department"]
        extracted_person: str = f"{extracted_text} works in {extracted_department}"

        safe_text: str = pseudoanonymize(analyzer, anonymizer, extracted_person).text

        last_questions: str = "\n\n".join(history[-max_questions:])

        prompt_llm: str = f"""
                        You are an HR assistant. 
                        Answer ONLY using the provided context. If the answer is not in the context, say "I don't have enough information".
                        If the requested information is masked with tags like <PERSON> or <ID_NUM> , say "I don't share this information". 
                        Provide ONLY the final answer. Do NOT explain your reasoning.

                        Context: {safe_text}
                        User query: {question}
                        Conversation history:{last_questions}
                        """

        answer_ai: Any = llm.invoke(prompt_llm)
        print(f"{answer_ai.content}")
        history.append(f"User: {question}\nAI: {answer_ai.content}")


def main() -> None:
    active_id: int = user_login()

    # Preparing analyze and anonymize engine
    analyzer: AnalyzerEngine = AnalyzerEngine()
    pesel_recognition(analyzer)
    anonymizer: AnonymizerEngine = AnonymizerEngine()
    my_embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_base: Chroma = Chroma(persist_directory="base_hr", embedding_function=my_embedding)

    # Api key in other file
    try:
        with open("../token.txt", "r") as f:
            api_key: str = f.read().strip()
    except FileNotFoundError:
        print("File not found")
        return

    os.environ["GROQ_API_KEY"] = api_key
    # Choosing llm model
    llm: ChatGroq = ChatGroq(model_name="openai/gpt-oss-120b")

    # Question loop
    user_question(active_id, vector_base, analyzer, anonymizer, llm)

if __name__ == '__main__':
    main()
