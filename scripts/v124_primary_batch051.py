"""Primary examination of v122 internal candidate states 51–60."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (51, "Which 1994 film starred actor Danny Trejo", "quarantine_pending_repair", ["question_year_contradicts_retrieved_film", "query_copies_visible_1995_list"],
  "The visible Trejo biography lists Heat as a 1995 film, not 1994. The copied list retrieves another article also calling Heat a 1995 film. Search did not find evidence for the question's year; the target pair needs a fresh 1995 wording or different 1994 film.", None,
  [("before_action", "Danny Trejo", "\"Heat\" (1995)"), ("retrieved_after_search", "Heat (1995 film)", "Heat is a 1995 American crime film")]),
 (52, "Which 1994 film starred actor Danny Trejo", "quarantine_pending_repair", ["answer_contradicts_question_year"],
  "The answer Heat contradicts the 1994 year in the question: both Trejo's film list and the returned Heat article date it to 1995. Do not treat a close year as equivalent.", None,
  [("before_action", "Danny Trejo", "\"Heat\" (1995)"), ("before_action", "Heat (1995 film)", "Heat is a 1995 American crime film")]),
 (53, "Which Brazilian actress starred", "retain_candidate", ["query_copies_visible_cast"],
  "The film article lists Alice Braga in the cast but gives no nationality. Its copied sentence retrieves Braga's biography, explicitly identifying her as a Brazilian actress.", None,
  [("before_action", "Journey to the End of the Night (film)", "Alice Braga"), ("retrieved_after_search", "Alice Braga", "is a Brazilian actress")]),
 (54, "Which Brazilian actress starred", "retain_candidate", [],
  "The Eric Eason film lists Alice Braga among its stars, and her biography calls her a Brazilian actress. The answer is supported.", None,
  [("before_action", "Journey to the End of the Night (film)", "directed by Eric Eason"), ("before_action", "Journey to the End of the Night (film)", "Alice Braga"), ("before_action", "Alice Braga", "is a Brazilian actress")]),
 (55, "Who was born in 1978", "retain_candidate", ["query_tests_visible_birth_year_candidate"],
  "The initial James Ponsoldt article supplies 1978 and names his film The Circle, but does not yet connect that film to Dave Eggers. Searching the visible candidate name retrieves The Circle article, verifying the Eggers writing relation. This is a grounded candidate test.", None,
  [("before_action", "James Ponsoldt", "born 1978"), ("before_action", "James Ponsoldt", "\"The Circle\" (2017)"), ("retrieved_after_search", "The Circle (2017 film)", "written by Ponsoldt and Dave Eggers")]),
 (56, "Who was born in 1978", "retain_candidate", [],
  "Ponsoldt's biography states his birth year and The Circle directorship; the film article identifies him as co-writer with Dave Eggers. The answer is supported.", None,
  [("before_action", "James Ponsoldt", "born 1978"), ("before_action", "James Ponsoldt", "\"The Circle\" (2017)"), ("before_action", "The Circle (2017 film)", "written by Ponsoldt and Dave Eggers")]),
 (57, "What was the job of the Patricia Franca's character", "quarantine_pending_repair", ["no_character_job_in_evidence", "query_copies_plot_not_job"],
  "The Orfeu film article names Patrícia França's character Eurídice but gives no occupation. The retrieved mythology article mentions Apollo driving the sun chariot as a parent of Eurydice in one tradition; it does not describe Eurydice's job or the film character's occupation.", None,
  [("before_action", "Orfeu", "Patrícia França as Eurídice"), ("retrieved_after_search", "Eurydice", "Apollo (the god of music, prophecy, and light, who also drove the sun chariot")]),
 (58, "What was the job of the Patricia Franca's character", "quarantine_pending_repair", ["answer_transfers_apollo_attribute_to_eurydice", "no_character_job_in_evidence"],
  "'Drove the sun chariot' is attached to Apollo in a parenthetical description, not to Eurydice, and no visible film source gives her occupation. The answer is a clear relation-splice error and must be excluded.", None,
  [("before_action", "Orfeu", "Patrícia França as Eurídice"), ("before_action", "Eurydice", "Apollo (the god of music, prophecy, and light, who also drove the sun chariot")]),
 (59, "What part did The King of Hollywood", "retain_candidate", ["query_copies_visible_film_cast"],
  "The film article says Clark Gable played a brave sea captain. Copying that sentence retrieves Gable's biography, confirming 'King of Hollywood' is his nickname.", None,
  [("before_action", "China Seas (film)", "Clark Gable as a brave sea captain"), ("retrieved_after_search", "Clark Gable", "referred to as \"The King of Hollywood\"")]),
 (60, "What part did The King of Hollywood", "retain_candidate", [],
  "Clark Gable is the person called the King of Hollywood, and the film article describes his part as a brave sea captain. The answer is supported.", None,
  [("before_action", "Clark Gable", "referred to as \"The King of Hollywood\""), ("before_action", "China Seas (film)", "Clark Gable as a brave sea captain")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-051.jsonl", cohort="v122_internal_candidates")
