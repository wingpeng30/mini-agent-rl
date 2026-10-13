"""Primary examination of v122 internal candidate states 321–326."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (321, "Les Brewer co-founded a college", "retain_candidate", ["query_uses_visible_college_relation", "initial_island_not_town"],
  "The initial Les Brewer article identifies the College of the Atlantic, but gives Mount Desert Island rather than a town. Searching that college relation retrieves the college article, which identifies Bar Harbor as its town. The query is grounded despite copying a long sentence.", None,
  [("before_action", "Les Brewer", "co-founded the College of the Atlantic"), ("retrieved_after_search", "College of the Atlantic", "located in Bar Harbor on Mount Desert Island")]),
 (322, "Les Brewer co-founded a college", "retain_candidate", [],
  "Les Brewer's biography names the College of the Atlantic and that college's article places it in Bar Harbor. The city/town answer is directly supported by the two documents.", None,
  [("before_action", "Les Brewer", "co-founded the College of the Atlantic"), ("before_action", "College of the Atlantic", "located in Bar Harbor on Mount Desert Island")]),
 (323, "What year was the narrator of the British television series", "retain_candidate", ["query_uses_visible_narrator_link"],
  "The initial The Cruise article identifies Hugh Bonneville as narrator, and retrieval returns his biography with 1963 as birth year. The query follows the visible missing-person link.", None,
  [("before_action", "The Cruise (2016 TV series)", "narrated by Hugh Bonneville"), ("retrieved_after_search", "Hugh Bonneville", "born 10 November 1963")]),
 (324, "What year was the narrator of the British television series", "retain_candidate", [],
  "The Cruise narrator is Hugh Bonneville, whose biography states a 1963 birth date. The answer year is supported by the action-time documents.", None,
  [("before_action", "The Cruise (2016 TV series)", "narrated by Hugh Bonneville"), ("before_action", "Hugh Bonneville", "born 10 November 1963")]),
 (325, "Where was a Foster's heavy beer originated", "repair_candidate", ["anachronistic_group_brand_relation", "answer_visible_before_search", "question_grammar"],
  "The initial Brisbane Bitter article already states Brisbane, Australia and a 1970s introduction by Carlton & United Breweries. Retrieval describes Carlton & United Breweries as a Foster's Group subsidiary, but also says the Foster's Group naming occurred in 1990; the question's Foster's ownership framing is therefore anachronistic for origin in the 1970s. Searching the introduction sentence is redundant for location.",
  "Ask where Brisbane Bitter, the Carlton & United Breweries heavy beer introduced in the 1970s, originated; answer directly and replay. Do not describe its 1970s origin as Foster's-branded without period evidence.",
  [("before_action", "Brisbane Bitter", "heavy beer originating, as the name suggests, in Brisbane, Australia"), ("before_action", "Brisbane Bitter", "introduced by Carlton & United Breweries in the 1970s"), ("retrieved_after_search", "Carlton &amp; United Breweries", "In 1990, Elders IXL was renamed as Foster's Group")]),
 (326, "Where was a Foster's heavy beer originated", "repair_candidate", ["paired_anachronistic_group_brand_relation"],
  "Brisbane, Australia is the stated origin of Brisbane Bitter. The original question instead calls the 1970s beer Foster's, relying on a later group relationship rather than source evidence that this was its identity at origin.",
  "Keep the location only in a corrected Brisbane Bitter/Carlton & United Breweries question after replay.",
  [("before_action", "Brisbane Bitter", "originating, as the name suggests, in Brisbane, Australia"), ("before_action", "Carlton &amp; United Breweries", "In 1990, Elders IXL was renamed as Foster's Group")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-078.jsonl", cohort="v122_internal_candidates")
