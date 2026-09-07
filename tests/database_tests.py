import pytest
from langchain_core.documents import Document
from src.create_database import create_users


def test_create_users_logic():
    # Setup test data
    doc_base = []
    start_id = 1
    names = ['Adam']
    surnames = ['Nowak']
    departments = ['IT']

    # Execute the function
    new_id = create_users(names, surnames, departments, start_id, doc_base)

    # Verify the outcomes
    assert new_id == 2, "ID should increment by 1"
    assert len(doc_base) == 1, "Exactly one document should be created"

    doc = doc_base[0]
    assert isinstance(doc, Document), "The object must be an instance of LangChain Document"

    # Verify metadata
    assert doc.metadata["user_id"] == 1, "Incorrect user_id assignment in metadata"
    assert doc.metadata["department"] == "IT", "Incorrect department assignment in metadata"

    # Verify page content structure
    assert "Adam" in doc.page_content, "Name is missing from the document content"
    assert "Nowak" in doc.page_content, "Surname is missing from the document content"
    assert "PESEL" in doc.page_content, "The word 'PESEL' is missing from the document content"