"""Primary examination of v122 internal candidate states 221–230."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (221, "Who did the actress who played Emma", "retain_candidate", ["query_copies_visible_role_relation"],
  "The Edge of Darkness snippet identifies Joanne Whalley as Emma. The search follows that visible role and retrieves her biography, which states she married Val Kilmer in 1988.", None,
  [("before_action", "Edge of Darkness", "Emma (played by Joanne Whalley)"), ("retrieved_after_search", "Joanne Whalley", "marriage to Val Kilmer in 1988")]),
 (222, "Who did the actress who played Emma", "retain_candidate", [],
  "Joanne Whalley played Emma, and her biography directly records marriage to Val Kilmer in 1988. The answer is supported.", None,
  [("before_action", "Edge of Darkness", "Emma (played by Joanne Whalley)"), ("before_action", "Joanne Whalley", "marriage to Val Kilmer in 1988")]),
 (223, "What team, which in German means", "repair_candidate", ["german_name_called_meaning"],
  "The initial Austria team article presents Österreichische Fußballnationalmannschaft as its German name, not an English phrase that 'means' that German string. The search retrieves Group D and confirms Austria as a member. The answer is supported, but the language relation in the question is reversed.",
  "Ask which team has the German name Österreichische Fußballnationalmannschaft and competed in UEFA Group D; replay.",
  [("before_action", "Austria national football team", "German: \"Österreichische Fußballnationalmannschaft\""), ("retrieved_after_search", "2018 FIFA World Cup qualification – UEFA Group D", "Wales, Austria, Serbia")]),
 (224, "What team, which in German means", "repair_candidate", ["paired_german_name_called_meaning"],
  "The Austria national football team matches both the German-name snippet and Group D membership. The original 'which in German means' wording should be revised to 'whose German name is'.",
  "Retain the team name only with revised, replayed wording.",
  [("before_action", "Austria national football team", "German: \"Österreichische Fußballnationalmannschaft\""), ("before_action", "2018 FIFA World Cup qualification – UEFA Group D", "Wales, Austria, Serbia")]),
 (225, "What time frame is covered by the document", "retain_candidate", ["query_uses_visible_diary_title"],
  "The initial diary article says Miep Gies recovered Anne Frank's diary and gave it to her father. The query targets that visible title and retrieves Anne Frank's biography with the 1942–1944 hiding period.", None,
  [("before_action", "The Diary of a Young Girl", "diary was retrieved by Miep Gies, who gave it to Anne's father"), ("retrieved_after_search", "Anne Frank", "life in hiding from 1942 to 1944")]),
 (226, "What time frame is covered by the document", "retain_candidate", [],
  "The document is Anne Frank's diary given by Gies to Otto Frank; the retrieved author biography dates the hiding documented in it to 1942–1944. The answer is supported.", None,
  [("before_action", "The Diary of a Young Girl", "gave it to Anne's father, Otto Frank"), ("before_action", "Anne Frank", "life in hiding from 1942 to 1944")]),
 (227, "Which softcore porn magazine featured photos", "repair_candidate", ["photos_stronger_than_featured_wrestlers"],
  "The wrestling article says its wrestlers were featured in Penthouse, Playboy and Maxim; it does not explicitly state that photos of the league appeared. The retrieved Penthouse article confirms a softcore-pictorial magazine, but not this specific photographic content. The answer identity is plausible yet the question overstates the documented feature.",
  "Ask which softcore-pictorial magazine was listed among magazines that featured NWWL wrestlers; replay.",
  [("before_action", "Naked Women's Wrestling League", "wrestlers have been featured in magazines such as \"Penthouse\", \"Playboy\", and \"Maxim\""), ("retrieved_after_search", "Penthouse (magazine)", "softcore pornographic pictorials")]),
 (228, "Which softcore porn magazine featured photos", "repair_candidate", ["paired_unverified_photos_relation"],
  "Penthouse is in the list of magazines featuring wrestlers and is described as softcore, but the supplied relation does not verify the question's 'photos of the league' claim.",
  "Retain Penthouse only under a feature-based, replayed question with independent review.",
  [("before_action", "Naked Women's Wrestling League", "wrestlers have been featured in magazines such as \"Penthouse\""), ("before_action", "Penthouse (magazine)", "softcore pornographic pictorials")]),
 (229, "What former National Security Advisor chaired", "quarantine_pending_repair", ["inflammatory_organization_purpose_unverified", "answer_already_visible_before_search", "unnecessary_search"],
  "The sole organization snippet itself directly names Zbigniew Brzezinski and makes a politically charged claim that the peace-named committee promoted an anti-Russian Wahhabi insurgency. The retrieved biography verifies his National Security Advisor role but does not corroborate that purpose. The query repeats the already-visible chair line. This item should not propagate the charged organization characterization on this evidence alone.",
  "Quarantine pending independent primary-source corroboration of the organization's stated purpose; a neutral chairmanship question would be a different item and still needs review.",
  [("before_action", "American Committee for Peace in Chechnya", "dedicated to promoting the anti-Russian Wahhabi insurgency"), ("before_action", "American Committee for Peace in Chechnya", "Chaired by former National Security Advisor Zbigniew Brzezinski"), ("retrieved_after_search", "Zbigniew Brzezinski", "National Security Advisor from 1977 to 1981")]),
 (230, "What former National Security Advisor chaired", "quarantine_pending_repair", ["paired_inflammatory_purpose_unverified", "paired_unnecessary_search"],
  "Brzezinski is named as chair in the initial snippet and as a former National Security Advisor in his biography. The question also asserts a serious organizational purpose supported only by the single uncorroborated snippet, and the search was unnecessary for the chair name.",
  "Keep the frozen pair quarantined; corroborate the purpose independently or formulate a neutral, separately reviewed question.",
  [("before_action", "American Committee for Peace in Chechnya", "dedicated to promoting the anti-Russian Wahhabi insurgency"), ("before_action", "American Committee for Peace in Chechnya", "Chaired by former National Security Advisor Zbigniew Brzezinski"), ("before_action", "Zbigniew Brzezinski", "National Security Advisor from 1977 to 1981")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-068.jsonl", cohort="v122_internal_candidates")
