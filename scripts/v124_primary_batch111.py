"""Primary examination of v123 internal candidate states 101–110."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (101, "In which state did the event take place when Cody Garbrandt", "retain_candidate", ["query_uses_visible_ufc207", "retrieval_adds_state"],
  "The initial Garbrandt article identifies UFC 207 as the event at which he defeated Dominick Cruz. The query quotes that established link and retrieves the UFC 207 article locating the arena in Paradise, Nevada.", None,
  [("before_action", "Cody Garbrandt", "defeat Dominick Cruz at UFC 207"), ("retrieved_after_search", "UFC 207", "Paradise, Nevada")]),
 (102, "In which state did the event take place when Cody Garbrandt", "retain_candidate", [],
  "Garbrandt's victory is tied to UFC 207, whose retrieved article places the event in Nevada. The state answer is supported without using his unrelated birth or residence.", None,
  [("before_action", "Cody Garbrandt", "defeat Dominick Cruz at UFC 207"), ("before_action", "UFC 207", "Paradise, Nevada")]),
 (103, "Who was the goalkeeper who blocked", "repair_candidate", ["blocked_footballer_vague", "question_player_typo", "retrieval_specific_penalty_save"],
  "The initial Paul Scholes article calls him the most decorated English footballer. A name search retrieves the 2005 FA Cup Final account saying Jens Lehmann saved Scholes's penalty shot. The question's 'goalkeeper who blocked the footballer player' neither specifies the shot nor the final and repeats 'footballer player'.",
  "Ask which Arsenal goalkeeper saved Paul Scholes's shot in the 2005 FA Cup Final shoot-out, then replay.",
  [("before_action", "Paul Scholes", "most decorated English footballer of all time"), ("retrieved_after_search", "2005 FA Cup Final", "Paul Scholes' shot was saved by Arsenal goalkeeper Jens Lehmann")]),
 (104, "Who was the goalkeeper who blocked", "repair_candidate", ["paired_vague_block_relation", "answer_supported_for_specific_match"],
  "Jens Lehmann is explicitly credited with saving Paul Scholes's shot in the 2005 FA Cup Final. The frozen question lacks that event and action and contains a duplicated player noun; the answer is supported only for the specific match relation.",
  "Use a match- and shot-specific question and replay.",
  [("before_action", "Paul Scholes", "most decorated English footballer of all time"), ("before_action", "2005 FA Cup Final", "Paul Scholes' shot was saved by Arsenal goalkeeper Jens Lehmann")]),
 (105, "Vice Versa and Mothering were both what", "repair_candidate", ["vice_versa_work_ambiguous", "query_preselects_magazine"],
  "The initial input includes two Vice Versa films and a distinct Vice Versa magazine; Mothering is absent. The query explicitly chooses Vice Versa (magazine), and retrieval finds Mothering as a magazine. The bare title in the question does not identify which Vice Versa work is intended, so this search choice is not licensed by the wording.",
  "Specify Vice Versa (1947–1948 periodical) and Mothering magazine, or add a unique periodical clue, then replay.",
  [("before_action", "Vice Versa (1948 film)", "British comedy film"), ("before_action", "Vice Versa (1916 film)", "British silent fantasy film"), ("before_action", "Vice Versa (magazine)", "U.S. periodical published especially for lesbians"), ("retrieved_after_search", "Mothering (magazine)", "Mothering is a magazine")]),
 (106, "Vice Versa and Mothering were both what", "repair_candidate", ["paired_vice_versa_title_ambiguity", "answer_only_selected_work"],
  "Magazine applies to the selected Vice Versa periodical and Mothering, but the unqualified Vice Versa title also denotes visible films. The answer depends on a work selection supplied by the query rather than by the question.",
  "Disambiguate the Vice Versa periodical in a revised pair.",
  [("before_action", "Vice Versa (1948 film)", "British comedy film"), ("before_action", "Vice Versa (magazine)", "U.S. periodical"), ("before_action", "Mothering (magazine)", "Mothering is a magazine")]),
 (107, '"Calling All Girls" is a song', "retain_candidate", ["query_uses_visible_writer", "retrieval_adds_nationality"],
  "The initial song article names Roger Taylor as writer and associates the track with Queen. A query quoting that credit retrieves the Queen drummer's biography, which identifies him as an English musician; the band context disambiguates same-name people.", None,
  [("before_action", "Calling All Girls", "written by Roger Taylor"), ("before_action", "Calling All Girls", "British rock band Queen"), ("retrieved_after_search", "Roger Taylor (Queen drummer)", "English musician")]),
 (108, '"Calling All Girls" is a song', "retain_candidate", [],
  "Roger Taylor is credited as the song writer, and his Queen drummer biography identifies him as English. The answer is supported by an action-time, band-disambiguated chain.", None,
  [("before_action", "Calling All Girls", "written by Roger Taylor"), ("before_action", "Roger Taylor (Queen drummer)", "English musician")]),
 (109, "What Album was circulated to all KFC's in Indonesia", "quarantine_pending_repair", ["multiple_albums_satisfy_predicate", "two_part_question_answer_one_part", "query_uses_other_album"],
  "The initial input already gives The Best of BCL and Our Biggest Thing Ever as albums circulated in all Indonesian KFC stores; retrieval adds Terbaik and Timeless with the same claim. The question gives no unique album criterion and separately asks for KFC headquarters, but the frozen answer supplies only Timeless. The query even includes Our Biggest Thing Ever, a competing answer.",
  "Choose an album with a distinguishing title/artist/date and ask a single target, or ask explicitly for all qualifying albums plus Louisville; create a new grounded pair.",
  [("before_action", "KFC", "Headquartered in Louisville, Kentucky"), ("before_action", "The Best of BCL", "circulated in all KFC stores in Indonesia"), ("before_action", "Our Biggest Thing Ever", "circulated in all KFC stores in Indonesia"), ("retrieved_after_search", "Timeless (Ungu album)", "circulated in all KFC stores in Indonesia")]),
 (110, "What Album was circulated to all KFC's in Indonesia", "quarantine_pending_repair", ["paired_nonunique_album", "headquarters_unanswered"],
  "Timeless is one of multiple albums reported as circulated through Indonesian KFC stores. The frozen answer omits the second question part, KFC's Louisville headquarters, and the title is not uniquely selected by the wording.",
  "Do not use this answer state without a single, uniquely specified question and complete target.",
  [("before_action", "KFC", "Headquartered in Louisville, Kentucky"), ("before_action", "The Best of BCL", "circulated in all KFC stores in Indonesia"), ("before_action", "Our Biggest Thing Ever", "circulated in all KFC stores in Indonesia"), ("before_action", "Timeless (Ungu album)", "circulated in all KFC stores in Indonesia")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-111.jsonl", cohort="v123_internal_candidates")
