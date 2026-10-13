"""Primary examination of v122 training-supplement candidate states 11–20."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (11, "What was composed between 1861 and 1865", "retain_candidate", ["answer_visible_before_search", "query_confirms_composer_membership"],
  "The symphony title and 1861–65 dates are already visible. Searching its title retrieves Rimsky-Korsakov's biography, which confirms the composer belonged to The Five, so the search checks the question's qualifying relation rather than finding the title.", None,
  [("before_action", "Symphony No. 1 (Rimsky-Korsakov)", "composed his Symphony No. 1 in E minor"), ("retrieved_after_search", "Nikolai Rimsky-Korsakov", "member of the group of composers known as The Five")]),
 (12, "What was composed between 1861 and 1865", "retain_candidate", ["title_visible_before_paired_search"],
  "The work source identifies Symphony No. 1 and its 1861–65 composition under Balakirev; the retrieved biography confirms Rimsky-Korsakov's Russian composer and The Five status. The answer is supported, although its name was already visible.", None,
  [("before_action", "Symphony No. 1 (Rimsky-Korsakov)", "between 1861 and 1865 under the guidance of Mily Balakirev"), ("before_action", "Nikolai Rimsky-Korsakov", "Russian composer, and a member of the group of composers known as The Five")]),
 (13, "The Asura friederikeae moth", "retain_candidate", ["query_uses_visible_country", "search_checks_capital_qualifier"],
  "Asura friederikeae is initially said to occur in Kenya and Uganda. Searching that geographic sentence retrieves Kenya's article establishing Nairobi as its capital, selecting Kenya among the two countries on the asked qualifier.", None,
  [("before_action", "Asura friederikeae", "found in Kenya and Uganda"), ("retrieved_after_search", "Kenya", "capital and largest city is Nairobi")]),
 (14, "The Asura friederikeae moth", "retain_candidate", [],
  "The Asura article places the moth in Kenya and Uganda, and the Kenya article directly gives Nairobi as capital. Kenya is the supported answer. The distinct Exilisia friederikeae is not used as the primary taxon.", None,
  [("before_action", "Asura friederikeae", "found in Kenya and Uganda"), ("before_action", "Kenya", "capital and largest city is Nairobi")]),
 (15, "Real is a South Korean action thriller noir film", "retain_candidate", ["query_uses_visible_film_cast"],
  "The 2017 Real film names Choi Jin-ri in its cast. Searching that cast line retrieves her Sulli biography, which gives her full name and former f(x) membership. The unrelated real-estate result is ignored.", None,
  [("before_action", "Real (2017 film)", "starring Kim Soo-hyun, Sung Dong-il, Lee Sung-min, Choi Jin-ri and Jo Woo-jin"), ("retrieved_after_search", "Sulli", "former member of South Korean girl group f(x)")]),
 (16, "Real is a South Korean action thriller noir film", "retain_candidate", [],
  "Choi Jin-ri is named in the Real cast; the Sulli biography maps that name to a former member of f(x). The answer uses her legal name and is supported at the answer stage.", None,
  [("before_action", "Real (2017 film)", "Choi Jin-ri"), ("before_action", "Sulli", "Choi Jin-ri"), ("before_action", "Sulli", "former member of South Korean girl group f(x)")]),
 (17, "The Houses October Built was produced by the same producer", "repair_candidate", ["producer_created_relation_inflated", "query_copies_hits_list"],
  "The Houses October Built names Steven Schneider as producer and says his hits include Insidious. The retrieved Insidious article identifies the 2010 film and Leigh Whannell as writer, but these snippets do not establish that Schneider 'created' it. The wording promotes a loose hits association into a specific creator role.",
  "Ask for the writer of the 2010 film Insidious, listed among producer Steven Schneider's hits, only after an independent reviewer accepts that association; otherwise drop this pair.",
  [("before_action", "The Houses October Built", "Steven Schneider, whose hits include \"Paranormal Activity\", \"Insidious\""), ("retrieved_after_search", "Insidious (film)", "written by Leigh Whannell")]),
 (18, "The Houses October Built was produced by the same producer", "repair_candidate", ["paired_creator_role_unsupported"],
  "Leigh Whannell is explicitly the writer of Insidious. The source about The Houses October Built only lists Insidious among Steven Schneider's hits; it does not prove he created that film, as the question asserts.",
  "Use a relation-faithful question with the same evidence after replay and independent review, or exclude.",
  [("before_action", "The Houses October Built", "Steven Schneider, whose hits include"), ("before_action", "Insidious (film)", "written by Leigh Whannell")]),
 (19, "How many member teams are in the hockey league", "repair_candidate", ["unanchored_league_size", "query_uses_visible_league_link"],
  "Murray Williamson's article identifies the United States Hockey League. Retrieval says the league has 17 member teams, but the question supplies no snapshot year even though league membership changes. The query is grounded; the time-varying numerical target is not anchored.",
  "Date the league-size claim to the source snapshot if available, or choose a stable fact; then replay the pair.",
  [("before_action", "Murray Williamson", "played, coached and managed in the United States Hockey League"), ("retrieved_after_search", "United States Hockey League", "has 17 member teams")]),
 (20, "How many member teams are in the hockey league", "repair_candidate", ["paired_unanchored_league_size"],
  "Seventeen is stated for the USHL in the retrieved article, and Williamson is tied to that league. Without a date, the question presents a mutable count as timeless; source support for one snapshot does not fix that.",
  "Retain 17 only in a time-anchored, replayed version if provenance supplies its date.",
  [("before_action", "Murray Williamson", "United States Hockey League"), ("before_action", "United States Hockey League", "has 17 member teams")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-080.jsonl", cohort="v122_train_candidates")
