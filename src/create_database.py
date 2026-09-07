from langchain_core.documents import Document
from typing import List
import random
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_users(names: List[str], surnames: List[str], departments: List[str], current_id: int, docbase: List[Document]) -> int:
    """Creating imaginary user list"""
    for name in names:
        surname: str = random.choice(surnames)
        department: str = random.choice(departments)
        salary: int = random.randint(8000, 50000)
        pesel: str = ''.join(random.choices('0123456789', k=11))
        doc: Document = Document(
            page_content=f"{name} {surname}, PESEL {pesel}, earns {salary} PLN.",
            metadata={"user_id": current_id, "department": department}
        )
        docbase.append(doc)
        current_id += 1
    return current_id

m_names: List[str] = ['Adam', 'Aleksander', 'Antoni', 'Bartosz', 'Błażej', 'Bruno', 'Cezary', 'Damian', 'Daniel', 'Dawid', 'Dominik', 'Emil', 'Franciszek', 'Gabriel', 'Grzegorz', 'Igor', 'Jakub', 'Jan', 'Janusz', 'Jędrzej', 'Kacper', 'Karol', 'Konrad', 'Krzysztof', 'Leon', 'Maciej', 'Marcel', 'Marek', 'Mateusz', 'Michał', 'Mikołaj', 'Nikodem', 'Oskar', 'Patryk', 'Paweł', 'Piotr', 'Przemysław', 'Rafał', 'Robert', 'Szymon', 'Tadeusz', 'Tomasz', 'Wiktor', 'Witold', 'Wojciech', 'Xawery', 'Zbigniew', 'Zdzisław', 'Ziemowit', 'Łukasz']

f_names: List[str] = ['Aleksandra', 'Alicja', 'Amelia', 'Anna', 'Antonina', 'Barbara', 'Beata', 'Blanka', 'Bożena', 'Celina', 'Dominika', 'Edyta', 'Eliza', 'Emilia', 'Gabriela', 'Hanna', 'Helena', 'Iga', 'Ilona', 'Izabela', 'Jagoda', 'Julia', 'Justyna', 'Kaja', 'Kamila', 'Karolina', 'Katarzyna', 'Kinga', 'Klara', 'Laura', 'Lena', 'Liliana', 'Liwia', 'Magdalena', 'Maria', 'Martyna', 'Maja', 'Milena', 'Natalia', 'Nina', 'Oliwia', 'Patrycja', 'Paulina', 'Renata', 'Roksana', 'Sara', 'Sylwia', 'Weronika', 'Wiktoria', 'Zofia']

m_surnames: List[str] = ['Nowak', 'Kowalski', 'Wiśniewski', 'Wójcik', 'Kowalczyk', 'Kamiński', 'Lewandowski', 'Zieliński', 'Szymański', 'Woźniak', 'Dąbrowski', 'Kozłowski', 'Jankowski', 'Mazur', 'Wojciechowski', 'Kwiatkowski', 'Krawczyk', 'Kaczmarek', 'Piotrowski', 'Grabowski', 'Zając', 'Pawłowski', 'Michalski', 'Król', 'Wieczorek', 'Jabłoński', 'Wróbel', 'Nowicki', 'Majewski', 'Olszewski', 'Stępień', 'Jaworski', 'Malinowski', 'Adamczyk', 'Dudek', 'Górski', 'Pawlikowski', 'Witkowski', 'Rutkowski', 'Walczak', 'Sikora', 'Baran', 'Michalak', 'Szewczyk', 'Ostrowski', 'Tomaszewski', 'Pietrzak', 'Marciniak', 'Włodarczyk', 'Borkowski']

f_surnames: List[str] = ['Nowak', 'Kowalska', 'Wiśniewska', 'Wójcik', 'Kowalczyk', 'Kamińska', 'Lewandowska', 'Zielińska', 'Szymańska', 'Woźniak', 'Dąbrowska', 'Kozłowska', 'Jankowska', 'Mazur', 'Wojciechowska', 'Kwiatkowska', 'Krawczyk', 'Kaczmarek', 'Piotrowska', 'Grabowska', 'Zając', 'Pawłowska', 'Michalska', 'Król', 'Wieczorek', 'Jabłońska', 'Wróbel', 'Nowicka', 'Majewska', 'Olszewska', 'Stępień', 'Jaworska', 'Malinowska', 'Adamczyk', 'Dudek', 'Górska', 'Pawlikowska', 'Witkowska', 'Rutkowska', 'Walczak', 'Sikora', 'Baran', 'Michalak', 'Szewczyk', 'Ostrowska', 'Tomaszewska', 'Pietrzak', 'Marciniak', 'Włodarczyk', 'Borkowska']

departments: List[str] = ['PR', 'HR', 'IT']

ID: int = 1

doc_base: List[Document] = []

ID = create_users(m_names, m_surnames, departments, ID, doc_base)
ID = create_users(f_names, f_surnames, departments, ID, doc_base)

# Choosing embedding engine
my_embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Creating vector database
vector_base: Chroma = Chroma.from_documents(
    documents=doc_base,
    embedding=my_embedding,
    persist_directory="./base_hr"
)