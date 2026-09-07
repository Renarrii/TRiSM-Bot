from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from src.main import pesel_recognition, pseudoanonymize


def test_pseudoanonymize_masks_sensitive_data():
    # Setup Presidio engines with custom recognizers
    analyzer = AnalyzerEngine()
    pesel_recognition(analyzer)
    anonymizer = AnonymizerEngine()

    # Vulnerable test string
    vul_text = "Jan Kowalski, PESEL 12345678901, works in IT."

    # Masking
    safe_result = pseudoanonymize(analyzer, anonymizer, vul_text)
    safe_text = safe_result.text

    # Verify data protection and anonymization logic

    # 1. Ensure real PII is completely removed
    assert "Jan" not in safe_text, "Data leak: Name was not masked"
    assert "Kowalski" not in safe_text, "Data leak: Surname was not masked"
    assert "12345678901" not in safe_text, "CRITICAL: PESEL number leaked!"

    # 2. Ensure correct placeholder tags were inserted
    assert "<PERSON>" in safe_text, "Presidio failed to insert the <PERSON> tag"
    assert "<ID_NUM>" in safe_text, "Presidio failed to insert the <ID_NUM> tag"

    # 3. Ensure non-sensitive context remains untouched
    assert "works in IT" in safe_text, "Non-sensitive contextual data was improperly removed"
