"""Primary examination of exposed historical internal states 61–70."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (61, "How long was the male co-writer", "retain_candidate", ["query_truncates_visible_author_name", "exposed_development_set"],
  "The novel article names James Ward and Anne K. Brown. The query truncates the sentence after Anne K., but retrieval supplies James M. Ward's TSR employment duration.", None,
  [("before_action", "Pools of Darkness (novel)", "written by James Ward and Anne K. Brown"), ("retrieved_after_search", "Jim Ward (game designer)", "worked for more than 20 years")]),
 (62, "How long was the male co-writer", "retain_candidate", ["exposed_development_set"],
  "The novel names James Ward as co-author and Jim Ward's biography says he worked for TSR for more than 20 years. The answer preserves the 'more than' qualifier.", None,
  [("before_action", "Pools of Darkness (novel)", "written by James Ward and Anne K. Brown"), ("before_action", "Jim Ward (game designer)", "worked for more than 20 years")]),
 (63, "Enkyo Pat O'Hara is", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "Her biography identifies NYU's Tisch School but lacks the founding year. The copied employment sentence retrieves the school's article with 1965.", None,
  [("before_action", "Enkyo Pat O'Hara", "New York University's Tisch School of the Arts"), ("retrieved_after_search", "Tisch School of the Arts", "Founded on August 17, 1965")]),
 (64, "Enkyo Pat O'Hara is", "retain_candidate", ["exposed_development_set"],
  "The professor relation targets Tisch School of the Arts; its article says it was founded in 1965. The answer does not confuse another educator's birth year.", None,
  [("before_action", "Enkyo Pat O'Hara", "New York University's Tisch School of the Arts"), ("before_action", "Tisch School of the Arts", "Founded on August 17, 1965")]),
 (65, "The PARIS project is", "repair_candidate", ["question_misstates_staff_project_as_project_for_website", "query_copies_visible_context", "exposed_development_set"],
  "The source calls PARIS a privately organized project undertaken by staff members of The Register, not a project 'for a website'. The query is grounded and retrieves the site's publisher, but the question relation needs correction.", None,
  [("before_action", "Paper Aircraft Released Into Space", "privately organised endeavour undertaken by various staff members"), ("before_action", "Paper Aircraft Released Into Space", "web site \"The Register\""), ("retrieved_after_search", "The Register", "Situation Publishing Ltd is listed as the site's publisher")]),
 (66, "The PARIS project is", "repair_candidate", ["question_misstates_staff_project_as_project_for_website", "exposed_development_set"],
  "Situation Publishing Ltd is supported as The Register's publisher. The original question inaccurately implies the PARIS project was for the website, so the pair requires a wording repair before reuse.", None,
  [("before_action", "Paper Aircraft Released Into Space", "undertaken by various staff members"), ("before_action", "The Register", "Situation Publishing Ltd is listed as the site's publisher")]),
 (67, "What Japanese mixed martial artist", "retain_candidate", ["query_repeats_visible_title", "exposed_development_set"],
  "The initial Yamamoto article names Kevin Randleman as an opponent, and the team article places Randleman with Hammer House. A title-repeat search retrieves Randleman's biography verifying former UFC Heavyweight Champion status.", None,
  [("before_action", "Kenichi Yamamoto (mixed martial artist)", "taking on Kevin Randleman"), ("before_action", "Team Hammer House", "Kevin Randleman"), ("retrieved_after_search", "Kevin Randleman", "former UFC Heavyweight Champion")]),
 (68, "What Japanese mixed martial artist", "retain_candidate", ["exposed_development_set"],
  "Yamamoto's biography says he fought Randleman; Randleman's biography confirms the championship and past Hammer House association. The answer Kenichi Yamamoto is supported.", None,
  [("before_action", "Kenichi Yamamoto (mixed martial artist)", "taking on Kevin Randleman"), ("before_action", "Kevin Randleman", "former UFC Heavyweight Champion"), ("before_action", "Kevin Randleman", "previously associated with Mark Coleman's Team Hammer House")]),
 (69, "When was the British sitcom", "retain_candidate", ["query_uses_question_and_visible_cast", "exposed_development_set"],
  "The actress article lists several shows without indicating which was a Channel 4 sitcom. The question-plus-article query retrieves Catastrophe, explicitly identifying its broadcaster and first broadcast date.", None,
  [("before_action", "Margaret Cabourn-Smith", "appearances on \"Catastrophe\""), ("retrieved_after_search", "Catastrophe (2015 TV series)", "first broadcast on 19 January 2015 on Channel 4")]),
 (70, "When was the British sitcom", "retain_candidate", ["exposed_development_set"],
  "The actress article lists Catastrophe; its article identifies it as the British Channel 4 sitcom first broadcast on 19 January 2015. The answer is supported.", None,
  [("before_action", "Margaret Cabourn-Smith", "appearances on \"Catastrophe\""), ("before_action", "Catastrophe (2015 TV series)", "first broadcast on 19 January 2015 on Channel 4")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-042.jsonl", cohort="v117_exposed_internal50")
