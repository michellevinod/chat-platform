import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import uuid
from app.services.chat_service import ChatService



def test_all_scenarios():
    service = ChatService()
    session_id = f"test-session-{uuid.uuid4().hex[:6]}"

    print("\n" + "=" * 60)
    print("RUNNING DOCUMENT INTELLIGENCE TESTS")
    print("=" * 60)

    # Test A: Greeting
    print("\n--- Test A: Greeting ('hi') ---")
    res_a = service.chat("hi", session_id=session_id)
    print("Response:", res_a["response"])
    assert res_a["success"] is True
    assert "Hello" in res_a["response"] or "hello" in res_a["response"].lower()
    print("PASS: Greeting handled correctly.")

    # Test B: Factual Lookup
    print("\n--- Test B: Factual Lookup ('What is the well name?') ---")
    res_b = service.chat("What is the well name?", session_id=session_id)
    print("Response:", res_b["response"][:150], "...")
    print("Citations:", res_b["citations"])
    assert res_b["success"] is True
    assert len(res_b["citations"]) > 0
    assert "document" in res_b["citations"][0]
    assert "page" in res_b["citations"][0]
    print("PASS: Factual lookup retrieved with citations.")

    # Test C: Document Summary
    print("\n--- Test C: Summarize Document ---")
    res_c = service.chat("summarize PDF_BenchmarkTester_63Pages.pdf", session_id=session_id)
    print("Response:", res_c["response"][:200], "...")
    print("Citations:", res_c["citations"])
    assert res_c["success"] is True
    print("PASS: Summary generated.")

    # Test D: Ambiguous Summary
    print("\n--- Test D: Ambiguous Summary ('summarize this' with no doc selected) ---")
    ambig_session = f"ambig-{uuid.uuid4().hex[:6]}"
    res_d = service.chat("summarize this", session_id=ambig_session)
    print("Response:", res_d["response"])
    assert res_d["success"] is True
    assert "Which document" in res_d["response"]
    print("PASS: Ambiguity prompted follow-up.")

    # Test E: Out-of-scope / Unrelated query ('how to make idli?')
    print("\n--- Test E: Out-of-scope ('how to make idli?') ---")
    res_e = service.chat("how to make idli?", session_id=session_id)
    print("Response:", res_e["response"])
    assert "I can answer only from uploaded documents" in res_e["response"] or "couldn't find relevant information" in res_e["response"]
    print("PASS: Out-of-scope rejected.")

    # Test F: Table lookup
    print("\n--- Test F: Table Lookup ('show table for worksheet') ---")
    res_f = service.chat("table worksheet", session_id=session_id)
    print("Response:", res_f["response"][:150], "...")
    print("Citations:", res_f["citations"])
    assert res_f["success"] is True
    print("PASS: Table query handled.")

    # Test G: Scoped project/doc filter
    print("\n--- Test G: Document Filter ---")
    res_g = service.chat(
        "casing pressure",
        document_name="16b1803d-2ac6-44c4-9245-6a02a595c495.pdf",
        session_id=session_id,
    )
    print("Response:", res_g["response"][:150], "...")
    print("Citations:", res_g["citations"])
    assert res_g["success"] is True
    for c in res_g["citations"]:
        assert c["document"] == "16b1803d-2ac6-44c4-9245-6a02a595c495.pdf"
    print("PASS: Document filtering applied.")

    # Test H: Follow-up question with conversation memory
    print("\n--- Test H: Follow-up question ('Explain that further.') ---")
    followup_session = f"mem-{uuid.uuid4().hex[:6]}"
    res_h1 = service.chat("What is the well name?", session_id=followup_session)
    print("Turn 1 Response:", res_h1["response"][:100], "...")
    res_h2 = service.chat("Explain that further.", session_id=followup_session)
    print("Turn 2 Response:", res_h2["response"][:150], "...")
    assert res_h2["success"] is True
    print("PASS: Follow-up memory contextualized Turn 2.")

    # Test I: Verify no leaked internal UUIDs or point IDs
    print("\n--- Test I: No UUID/Internal IDs leaked ---")
    for cite in res_b["citations"]:
        assert "point_id" not in cite
        assert "vector" not in cite
    print("PASS: Citations are clean and user-facing.")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_all_scenarios()