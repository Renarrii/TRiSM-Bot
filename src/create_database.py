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

m_names: List[str] = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles', 'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian', 'George', 'Edward', 'Ronald', 'Timothy', 'Jason', 'Jeffrey', 'Ryan']

f_names: List[str] = ['Mary', 'Patricia', 'Linda', 'Barbara', 'Elizabeth', 'Jennifer', 'Maria', 'Susan', 'Margaret', 'Dorothy', 'Lisa', 'Nancy', 'Karen', 'Betty', 'Helen', 'Sandra', 'Donna', 'Carol', 'Ruth', 'Sharon', 'Michelle', 'Laura', 'Sarah', 'Kimberly', 'Deborah', 'Jessica']

english_surnames: List[str] = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell', 'Carter', 'Roberts']

m_surnames: List[str] = english_surnames.copy()
f_surnames: List[str] = english_surnames.copy()

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
