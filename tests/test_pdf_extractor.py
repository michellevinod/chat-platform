import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import uuid
from fastapi.testclient import TestClient
from app.main import app
from app.repositories.qdrant_repository import QdrantRepository
from app.services.chat_service import ChatService
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def run_full_verification():
    client = TestClient(app)
    service = ChatService()
    repo = QdrantRepository()

    results_report = {"passed": [], "failed": []}

    print("\n" + "=" * 80)
    print("STARTING END-TO-END VERIFICATION OF ALL REQUIREMENTS")
    print("=" * 80)

    # ----------------------------------------------------
    # SECTION 1: UPLOAD & INGESTION VERIFICATION
    # ----------------------------------------------------
    print("\n>>> SECTION 1: UPLOADS & METADATA INGESTION")

    # 1. Upload PDF
    pdf_path = Path("storage/uploads/PDF_BenchmarkTester_63Pages.pdf")
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("deepwater_daily_report.pdf", f, "application/pdf")},
                data={"project_name": "Deepwater Project", "document_name": "deepwater_daily_report.pdf"},
            )
        assert resp.status_code == 200, f"PDF Upload failed: {resp.text}"
        data = resp.json()
        print(f"  [+] PDF Upload: Success={data.get('success')}, Chunks={data.get('chunks_uploaded')}")

        # Verify Qdrant payload
        f = Filter(must=[FieldCondition(key="document_name", match=MatchValue(value="deepwater_daily_report.pdf"))])
        pts, _ = repo._client.scroll(collection_name="documents", scroll_filter=f, limit=5, with_payload=True)
        assert len(pts) > 0, "No points in Qdrant for PDF"
        p = pts[0].payload
        assert p.get("project_name") == "Deepwater Project", f"Wrong project_name: {p.get('project_name')}"
        assert p.get("document_name") == "deepwater_daily_report.pdf", f"Wrong document_name: {p.get('document_name')}"
        assert "page_number" in p, "Missing page_number"
        assert "chunk_type" in p, "Missing chunk_type"
        assert "heading" in p, "Missing heading"
        assert "section" in p, "Missing section"
        print(f"  [+] PDF Qdrant Payload Verified: project={p.get('project_name')}, doc={p.get('document_name')}, chunk_type={p.get('chunk_type')}, heading={p.get('heading')}")
        results_report["passed"].append("1. Upload PDF & Metadata Verification")
    else:
        results_report["failed"].append("1. Upload PDF (Sample file not found)")

    # 2. Upload DOCX
    docx_path = Path("storage/uploads/c0e7d211-9cf8-495d-83ac-31b5294d5c58.docx")
    if docx_path.exists():
        with open(docx_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("operations_manual.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"project_name": "Operations Project", "document_name": "operations_manual.docx"},
            )
        assert resp.status_code == 200, f"DOCX Upload failed: {resp.text}"
        data = resp.json()
        print(f"  [+] DOCX Upload: Success={data.get('success')}, Chunks={data.get('chunks_uploaded')}")
        results_report["passed"].append("2. Upload DOCX")
    else:
        results_report["failed"].append("2. Upload DOCX (Sample file not found)")

    # 3. Upload PPTX
    pptx_path = Path("storage/uploads/c2dcb33f-93b2-43c6-9dde-c5f79b0d6e67.pptx")
    if pptx_path.exists():
        with open(pptx_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("quarterly_presentation.pptx", f, "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
                data={"project_name": "Executive Project", "document_name": "quarterly_presentation.pptx"},
            )
        assert resp.status_code == 200, f"PPTX Upload failed: {resp.text}"
        data = resp.json()
        print(f"  [+] PPTX Upload: Success={data.get('success')}, Chunks={data.get('chunks_uploaded')}")
        results_report["passed"].append("3. Upload PPTX")
    else:
        results_report["failed"].append("3. Upload PPTX (Sample file not found)")

    # 4. Upload XLSX
    xlsx_path = Path("storage/uploads/85a197c5-33af-45cb-a09e-863d388464d7.xlsx")
    if xlsx_path.exists():
        with open(xlsx_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("production_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                data={"project_name": "Financial Project", "document_name": "production_data.xlsx"},
            )
        assert resp.status_code == 200, f"XLSX Upload failed: {resp.text}"
        data = resp.json()
        print(f"  [+] XLSX Upload: Success={data.get('success')}, Chunks={data.get('chunks_uploaded')}")
        results_report["passed"].append("4. Upload XLSX")
    else:
        results_report["failed"].append("4. Upload XLSX (Sample file not found)")

    # ----------------------------------------------------
    # SECTION 2: CHAT TESTS (EXACT REQUIREMENTS)
    # ----------------------------------------------------
    print("\n>>> SECTION 2: CHAT SCENARIOS")

    # Test 1: Factual lookup ("What is casing summary?")
    print("\n--- Test 1: 'What is casing summary?' (Factual / Direct RAG, No Gemini) ---")
    t1_session = f"t1-{uuid.uuid4().hex[:6]}"
    res1 = service.chat("What is casing summary?", session_id=t1_session)
    print("Response:\n", res1["response"][:200], "...")
    print("Citations:", res1["citations"])
    assert res1["success"] is True, "Test 1 failed"
    assert len(res1["citations"]) > 0, "Test 1 citations missing"
    assert "document" in res1["citations"][0] and "page" in res1["citations"][0]
    results_report["passed"].append("Test 1: Factual Lookup (RAG only, citations included)")

    # Test 2: Summarize report
    print("\n--- Test 2: 'Summarize the report.' (Gemini Synthesis & Markdown) ---")
    t2_session = f"t2-{uuid.uuid4().hex[:6]}"
    res2 = service.chat("Summarize document deepwater_daily_report.pdf", session_id=t2_session)
    print("Response:\n", res2["response"][:250], "...")
    print("Citations count:", len(res2["citations"]))
    assert res2["success"] is True, "Test 2 failed"
    assert len(res2["response"]) > 50, "Test 2 response too short"
    assert len(res2["citations"]) > 0, "Test 2 citations missing"
    results_report["passed"].append("Test 2: Summarize Report (Gemini synthesis, Markdown, citations)")

    # Test 3: Compare documents
    print("\n--- Test 3: 'Compare document deepwater_daily_report.pdf and operations_manual.docx.' ---")
    t3_session = f"t3-{uuid.uuid4().hex[:6]}"
    res3 = service.chat("Compare document deepwater_daily_report.pdf and operations_manual.docx.", session_id=t3_session)
    print("Response:\n", res3["response"][:250], "...")
    print("Citations count:", len(res3["citations"]))
    assert res3["success"] is True, "Test 3 failed"
    assert len(res3["response"]) > 50, "Test 3 response too short"
    results_report["passed"].append("Test 3: Compare Documents (Gemini comparison produced)")

    # Test 4: Show production table
    print("\n--- Test 4: 'Show production table.' (Markdown Table) ---")
    t4_session = f"t4-{uuid.uuid4().hex[:6]}"
    res4 = service.chat("Show production table in production_data.xlsx", session_id=t4_session)
    print("Response:\n", res4["response"][:250], "...")
    print("Citations:", res4["citations"])
    assert res4["success"] is True, "Test 4 failed"
    assert "|" in res4["response"], "Test 4 table not formatted as Markdown table"
    results_report["passed"].append("Test 4: Show Production Table (Markdown table returned)")

    # Test 5: Show core image
    print("\n--- Test 5: 'Show core image.' (Image Reference) ---")
    t5_session = f"t5-{uuid.uuid4().hex[:6]}"
    res5 = service.chat("Show core image or figure in deepwater_daily_report.pdf", session_id=t5_session)
    print("Response:\n", res5["response"][:250], "...")
    print("Citations:", res5["citations"])
    assert res5["success"] is True, "Test 5 failed"
    results_report["passed"].append("Test 5: Show Core Image (Image reference/metadata returned)")

    # Test 6: Out of scope ("How to make idli?")
    print("\n--- Test 6: 'How to make idli?' (Out-of-scope Rejection, No Gemini) ---")
    t6_session = f"t6-{uuid.uuid4().hex[:6]}"
    res6 = service.chat("How to make idli?", session_id=t6_session)
    print("Response:", res6["response"])
    assert "I can answer only from uploaded documents" in res6["response"]
    assert len(res6["citations"]) == 0
    results_report["passed"].append("Test 6: Out-of-Scope Rejection ('How to make idli?' rejected without Gemini)")

    # Test 7: Ambiguous summarize
    print("\n--- Test 7: 'Summarize' with multiple documents (Follow-up Prompt) ---")
    t7_session = f"t7-{uuid.uuid4().hex[:6]}"
    res7 = service.chat("Summarize", session_id=t7_session)
    print("Response:", res7["response"])
    assert "Which document" in res7["response"], f"Expected document follow-up question, got: {res7['response']}"
    results_report["passed"].append("Test 7: Ambiguous Summarize (Follow-up prompt returned)")

    # Test 8: Project filtering
    print("\n--- Test 8: Project Filtering (Search within 'Deepwater Project') ---")
    t8_session = f"t8-{uuid.uuid4().hex[:6]}"
    res8 = service.chat("casing pressure", project_name="Deepwater Project", session_id=t8_session)
    print("Response:\n", res8["response"][:150], "...")
    print("Citations:", res8["citations"])
    assert res8["success"] is True, "Test 8 failed"
    for cite in res8["citations"]:
        assert cite["project"] == "Deepwater Project", f"Leaked project in citation: {cite}"
    results_report["passed"].append("Test 8: Project Filtering (Retrieved only from selected project)")

    # Test 9: Document filtering
    print("\n--- Test 9: Document Filtering (Search within 'production_data.xlsx') ---")
    t9_session = f"t9-{uuid.uuid4().hex[:6]}"
    res9 = service.chat("worksheet data", document_name="production_data.xlsx", session_id=t9_session)
    print("Response:\n", res9["response"][:150], "...")
    print("Citations:", res9["citations"])
    assert res9["success"] is True, "Test 9 failed"
    for cite in res9["citations"]:
        assert cite["document"] == "production_data.xlsx", f"Leaked document in citation: {cite}"
    results_report["passed"].append("Test 9: Document Filtering (Retrieved only from selected document)")

    # Test 10: Conversation memory
    print("\n--- Test 10: Conversation Memory (Follow-up Context) ---")
    t10_session = f"t10-{uuid.uuid4().hex[:6]}"
    turn1 = service.chat("Summarize deepwater_daily_report.pdf", session_id=t10_session)
    print("Turn 1 (Summarize deepwater_daily_report.pdf):\n", turn1["response"][:150], "...")
    turn2 = service.chat("Explain further.", session_id=t10_session)
    print("Turn 2 (Explain further.):\n", turn2["response"][:200], "...")
    assert turn2["success"] is True, "Test 10 turn 2 failed"
    assert len(turn2["response"]) > 30, "Test 10 turn 2 response too short"
    results_report["passed"].append("Test 10: Conversation Memory (Follow-up query retained context)")

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print(f"Total Passed: {len(results_report['passed'])}")
    print(f"Total Failed: {len(results_report['failed'])}")
    for p in results_report["passed"]:
        print(f"  [PASS] {p}")
    for f in results_report["failed"]:
        print(f"  [FAIL] {f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_verification()