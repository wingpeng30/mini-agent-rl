"""Primary examination of v123 internal candidate states 71–80."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (71, "Go North starred the eldest son of which bodybuilder", "repair_candidate", ["elder_vs_eldest_overclaim", "bodybuilder_relation_not_in_excerpts"],
  "The film cast lists Patrick Schwarzenegger, and the retrieved biography calls him the elder son of Arnold Schwarzenegger. It does not establish the stronger eldest-son claim, nor does any current excerpt identify Arnold's bodybuilder role. The question combines two unsupported qualifiers.",
  "Use a source explicitly establishing both eldest and bodybuilder, or weaken the question to the supported Patrick-to-Arnold father relation and replay.",
  [("before_action", "Go North (film)", "Patrick Schwarzenegger"), ("retrieved_after_search", "Patrick Schwarzenegger", "elder son of Arnold Schwarzenegger and Maria Shriver")]),
 (72, "Go North starred the eldest son of which bodybuilder", "repair_candidate", ["paired_unsupported_eldest_and_bodybuilder_qualifiers"],
  "Arnold Schwarzenegger is named as Patrick's father, but the action-time documents only say Patrick is an elder son and do not substantiate the bodybuilder clue. The answer string alone does not make the complete question supported.",
  "Keep the father name only under a revised relation with verified qualifiers and replay.",
  [("before_action", "Go North (film)", "Patrick Schwarzenegger"), ("before_action", "Patrick Schwarzenegger", "elder son of Arnold Schwarzenegger and Maria Shriver")]),
 (73, "Who did the actress who played Emma", "retain_candidate", ["query_uses_visible_actress", "retrieval_adds_spouse_year"],
  "The initial Edge of Darkness article identifies Joanne Whalley as Emma. The query repeats that relation and retrieves Whalley's biography, which says she married Val Kilmer in 1988.", None,
  [("before_action", "Edge of Darkness", "Emma (played by Joanne Whalley)"), ("retrieved_after_search", "Joanne Whalley", "marriage to Val Kilmer in 1988")]),
 (74, "Who did the actress who played Emma", "retain_candidate", [],
  "Joanne Whalley is credited as Emma and the retrieved biography names Val Kilmer as the person she married in 1988. The answer is supported.", None,
  [("before_action", "Edge of Darkness", "Joanne Whalley"), ("before_action", "Joanne Whalley", "marriage to Val Kilmer in 1988")]),
 (75, "What team, which in German means", "repair_candidate", ["german_name_misdescribed_as_meaning", "group_membership_retrieved"],
  "The Austria team article gives Österreichische Fußballnationalmannschaft as its German name, not a German phrase 'meaning' the team. The query uses the visible Austria name and retrieves Group D membership, so the factual selection is sound but the language relation needs correction.",
  "Say 'whose German name is Österreichische Fußballnationalmannschaft' and replay.",
  [("before_action", "Austria national football team", 'German: "Österreichische Fußballnationalmannschaft"'), ("retrieved_after_search", "2018 FIFA World Cup qualification – UEFA Group D", "Wales, Austria, Serbia")]),
 (76, "What team, which in German means", "repair_candidate", ["paired_name_meaning_confusion"],
  "Austria is directly listed in UEFA Group D and its national-team article supplies the German team name. The answer is supported, but the frozen question incorrectly frames a German name as a meaning relation.",
  "Use a German-name question and re-audit the pair.",
  [("before_action", "Austria national football team", 'German: "Österreichische Fußballnationalmannschaft"'), ("before_action", "2018 FIFA World Cup qualification – UEFA Group D", "Wales, Austria, Serbia")]),
 (77, "Carrie Pilby stars which Irish actor and musician", "retain_candidate", ["query_uses_visible_cast", "retrieval_adds_nationality_music_role"],
  "The initial Carrie Pilby film page lists Colin O'Donoghue. A cast-list query retrieves his biography, which directly calls him an Irish actor and musician and supplies his full name.", None,
  [("before_action", "Carrie Pilby (film)", "Colin O'Donoghue"), ("retrieved_after_search", "Colin O'Donoghue", "Irish actor and musician")]),
 (78, "Carrie Pilby stars which Irish actor and musician", "retain_candidate", [],
  "Colin O'Donoghue appears in the film cast, and the retrieved biography gives his full name Colin Arthur O'Donoghue and the Irish actor/musician description. The answer is supported.", None,
  [("before_action", "Carrie Pilby (film)", "Colin O'Donoghue"), ("before_action", "Colin O'Donoghue", "Colin Arthur O'Donoghue"), ("before_action", "Colin O'Donoghue", "Irish actor and musician")]),
 (79, "What nationality was the female star of the film The Favour", "retain_candidate", ["query_uses_visible_female_cast", "retrieval_adds_nationality"],
  "The initial film article names Natasha Richardson among the stars. The film-cast query retrieves her biography calling her an English actress. The other named film stars are men, so the female-star clue is sufficiently targeted within the shown cast.", None,
  [("before_action", "The Favour, the Watch and the Very Big Fish", "Bob Hoskins, Jeff Goldblum and Natasha Richardson"), ("retrieved_after_search", "Natasha Richardson", "English stage and screen actress")]),
 (80, "What nationality was the female star of the film The Favour", "retain_candidate", [],
  "Natasha Richardson is the female actor in the listed film cast, and her biography directly identifies her as English. The answer is supported at the recorded action time.", None,
  [("before_action", "The Favour, the Watch and the Very Big Fish", "Natasha Richardson"), ("before_action", "Natasha Richardson", "English stage and screen actress")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-108.jsonl", cohort="v123_internal_candidates")
