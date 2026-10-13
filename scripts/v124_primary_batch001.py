"""Persist the primary agent's explicit, case-by-case examination of rows 1–10.

These are primary diagnostic judgments, not independent approvals or train labels.
"""
from scripts.v123_cpu_core import digest, readl, write, writel
from scripts.v124_forensic import OUT


def evidence_binding(row, lineage):
    """Bind a review to its actual post-search evidence source.

    The v117 cohorts have separately reconstructed lineage. Later cohorts carry
    their retrieval replay in the hashed inventory row itself.
    """
    key = row["record_id"]
    if key in lineage:
        bound = lineage[key]
        assert len(bound["matches"]) == 1
        return bound, bound["matches"][0]["retrieved_after_search"]
    assert row["cohort"] not in {"v117_train", "v117_exposed_internal50"}
    bound = {"kind": "inventory_retrieval", "record_id": key,
             "source_row_sha256": row["source_row_sha256"],
             "record_sha256": row["record_sha256"],
             "retrieved_after_search": row["retrieved_after_search"]}
    return bound, row["retrieved_after_search"]

# Each entry was written after inspecting that case's full visible documents.
SPECS = [
    (1, "Who is the current woman", "quarantine_pending_repair", ["unanchored_current_time", "missing_entity_role_link_before_search"],
     "The question has no as-of date. Prior documents name a current First Lady but do not link Nana Konadu to that office; the returned Nana biography supplies this missing link. Retrieval succeeds, but the earlier generic-current search and office-selection rationale are weak. Do not substitute a contemporary name or infer a date from unrelated distractor text.",
     "Establish source snapshot provenance before a date-anchored question repair; evaluate a query about Nana's former position using only the pre-action named entity. No repaired target is approved yet.",
     [("before_action", "First Lady of Ghana", "The current First Lady is Rebecca Akufo-Addo."),
      ("retrieved_after_search", "Nana Konadu Agyeman Rawlings", "was the First Lady of Ghana from 4 June 1979 to 24 September 1979")]),
    (2, "Which other occupation", "retain_candidate", [],
     "The film identifies Kristy McNichol in the exact requested role; the Kristy McNichol biography identifies the actress as a singer. The linked entity and other occupation are supported; unrelated German-history documents add noise but do not invalidate this answer.", None,
     [("before_action", "Summer of My German Soldier (film)", "It stars Kristy McNichol as a Jewish-American girl and Bruce Davison as the German prisoner of war whom she befriends."),
      ("before_action", "Kristy McNichol", "Christina Ann McNichol (born September 11, 1962) is an American actress and singer.")]),
    (3, "Name the British-American musical", "retain_candidate", [],
     "The film document supports studio, genre and book adaptation, while the actor document links the specified role and actor to the same film. Multiple documents jointly establish the answer without splicing their quotations.", None,
     [("before_action", "David Tomlinson", "fraudulent magician Professor Emelius Browne in \"Bedknobs and Broomsticks\""),
      ("before_action", "Bedknobs and Broomsticks", "It is based upon the books \"The Magic Bedknob; or, How to Become a Witch in Ten Easy Lessons\" (1943)")]),
    (4, "Southbridge Municipal Airport", "retain_candidate", ["answer_does_not_require_multiple_documents"],
     "The airport document explicitly names the damaging event and the event document corroborates the date and event description. The verbose question does not require a second hop to select the answer; retain as supported stopping behavior, not evidence of multihop skill.", None,
     [("before_action", "Southbridge Municipal Airport", "The airport sustained damage on June 1st from the 2011 New England tornado outbreak."),
      ("before_action", "2011 New England tornado outbreak", "On June 1, 2011, seven tornadoes appeared in Massachusetts' Connecticut River Valley and southern Maine")]),
    (5, "Who formed first, The Get Up Kids", "retain_candidate", ["query_can_be_more_focused_but_not_proven_wrong"],
     "The prior evidence supplies 1995 for The Get Up Kids but not the Moonbabies formation date. The query contains both question entities and the comparison request. It retrieves the missing Moonbabies date, 1997. Appending the already-known band is not by itself a semantic rejection.", None,
     [("before_action", "The Get Up Kids", "Formed in 1995"),
      ("retrieved_after_search", "Moonbabies (band)", "Moonbabies is a Swedish duo formed in 1997")]),
    (6, "Which person that recorded", "retain_candidate", [],
     "The album credits Jan Akkerman in the original recording lineup; his biography explicitly gives the requested birth date. The target is supported through an ordinary entity join.", None,
     [("before_action", "Focus Plays Focus", "the original line-up of Thijs van Leer, Jan Akkerman, Martin Dresden and Hans Cleuver."),
      ("before_action", "Jan Akkerman", "Jan Akkerman (born 24 December 1946) is a Dutch guitarist.")]),
    (7, "Are Hilaire Belloc and Rupert Brooke", "retain_candidate", ["query_can_be_more_focused_but_not_proven_wrong"],
     "Belloc's writer role is known before the action but Brooke is absent. The query explicitly names both and asks whether they are writers; retrieved Brooke evidence establishes that he wrote poetry. A narrower query might be efficient, but the actual action is grounded and fills the missing relation.", None,
     [("before_action", "Hilaire Belloc", "was an Anglo-French writer and historian."),
      ("retrieved_after_search", "Rupert Brooke", "was an English poet known for his idealistic war sonnets")]),
    (8, "Who wrote a work that described", "retain_candidate", ["scope_read_as_sovereign_monarch_in_source"],
     "The essay covers Europe until the age of Louis XIV and explicitly names Voltaire. The Louis XIV document states the longest recorded reign for a sovereign-country monarch in European history. Preserve this source-supported scope in the review; the answer is not based on the unrelated Dawn Penn hit.", None,
     [("before_action", "Essai sur les mœurs et l'esprit des nations", "is a work by the French writer, historian, and philosopher Voltaire"),
      ("before_action", "Essai sur les mœurs et l'esprit des nations", "until the dawn of the age of Louis XIV"),
      ("before_action", "Louis XIV of France", "the longest recorded of any monarch of a sovereign country in European history.")]),
    (9, "Matthew Lewis appears in Bluestone", "retain_candidate", [],
     "The programme document explicitly identifies BBC Three; the actor biography independently links the named actor to that same BBC Three series. This is supported answer extraction.", None,
     [("before_action", "Bluestone 42", "first broadcast on 5 March 2013 on BBC Three."),
      ("before_action", "Matthew Lewis (actor)", "the BBC Three comedy drama \"Bluestone 42\".")]),
    (10, "Are Hilaire Belloc and Rupert Brooke", "retain_candidate", [],
     "Both entities are visible: Belloc is explicitly a writer, Brooke a poet who wrote war sonnets. Poet entails writer for this occupational question. This is a supported comparison, not an unsupported inference from an absent occupation.", None,
     [("before_action", "Hilaire Belloc", "was an Anglo-French writer and historian."),
      ("before_action", "Rupert Brooke", "was an English poet known for his idealistic war sonnets")]),
]


def save_batch(specs, batch_name, cohort="v117_train"):
    records = {r["record_id"]: r for r in readl(OUT / "inventory-private.jsonl")}
    lineage = {r["record_id"]: r for r in readl(OUT / "lineage-private.jsonl")}
    decisions = []
    for number, prefix, disposition, issues, rationale, repair, citations in specs:
        key = f"{cohort}:{number:04}"
        row = records[key]
        assert row["question"].startswith(prefix)
        binding, retrieved = evidence_binding(row, lineage)
        bound = []
        for stage, title, quote in citations:
            assert stage in {"before_action", "retrieved_after_search"}
            documents = row["before_action"] if stage == "before_action" else retrieved
            docs = [d for d in documents if d["title"] == title and quote in d["text"]]
            assert len(docs) == 1, (key, title, quote)
            bound.append({"stage": stage, "document_id": docs[0]["id"], "quote": quote})
        decisions.append({"record_id": key, "record_sha256": row["record_sha256"],
                          "reviewer": "primary_agent", "independent_review": False,
                          "disposition": disposition, "issues": issues, "rationale": rationale,
                          "repair_proposal": repair, "citations": bound,
                          "lineage_binding_sha256": digest(binding),
                          "approved_training_label": False, "training_allowed": False})
    assert batch_name.startswith("primary-batch-") and batch_name.endswith(".jsonl") and "/" not in batch_name and "\\" not in batch_name
    path = OUT / batch_name
    if path.exists() and readl(path) != decisions:
        raise ValueError("existing_primary_batch_differs")
    if not path.exists():
        writel(path, decisions)
    all_decisions = [r for file in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(file)]
    reviewed = {r["record_id"] for r in all_decisions}
    assert len(reviewed) == len(all_decisions), "duplicate_primary_review"
    assert all(r["record_sha256"] == records[r["record_id"]]["record_sha256"] for r in all_decisions)
    remaining = [key for key in records if key not in reviewed]
    write(OUT / "progress.json", {"total_state_records": len(records), "primary_reviewed": len(reviewed),
          "primary_remaining": len(remaining), "repaired_double_approved": 0,
          "status": "in_progress", "next_record": remaining[0] if remaining else None, "training_allowed": False})
    print(f"Saved {len(decisions)} primary reviews with exact stage-bound citations; no independent approvals.")


if __name__ == "__main__":
    save_batch(SPECS, "primary-batch-001.jsonl")
