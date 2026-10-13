"""Primary examination of v122 internal candidate states 131–140."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (131, "The band who performs", "retain_candidate", ["query_copies_visible_band_relation"],
  "Sunny Sundae Smile names My Bloody Valentine as the band. The search follows that visible relation and retrieves the band's formation year, which is absent initially.", None,
  [("before_action", "Sunny Sundae Smile", "band My Bloody Valentine"), ("retrieved_after_search", "My Bloody Valentine (band)", "formed in Dublin in 1983")]),
 (132, "The band who performs", "retain_candidate", [],
  "The song is credited to My Bloody Valentine, whose band article states it formed in 1983. The year is supported.", None,
  [("before_action", "Sunny Sundae Smile", "band My Bloody Valentine"), ("before_action", "My Bloody Valentine (band)", "formed in Dublin in 1983")]),
 (133, "Ma'alefa'ak is a character", "quarantine_pending_repair", ["creator_of_related_character_substituted", "series_vs_character_creation_ambiguity"],
  "The initial Martian Manhunter article says Joseph Samachson created that superhero, not Ma'alefa'ak or a clearly defined comic series. Retrieval then explicitly says Ma'alefa'ak was created by writer John Ostrander. The frozen target confuses related characters and the question does not identify which series' creator it means.",
  "Exclude this pair. A new question must state whether it asks for Ma'alefa'ak's creator (John Ostrander) or for Martian Manhunter's creator, with a fully supported relation.",
  [("before_action", "Martian Manhunter", "Created by writer Joseph Samachson"), ("retrieved_after_search", "Ma'alefa'ak", "Created by writer John Ostrander")]),
 (134, "Ma'alefa'ak is a character", "quarantine_pending_repair", ["wrong_creator_for_target_character", "question_ambiguous_series_vs_character"],
  "Joseph Samachson is explicitly credited for Martian Manhunter, whereas Ma'alefa'ak's own article credits John Ostrander. The answer selects the adjacent character's creator and cannot be approved for the stated question without rewriting it.",
  "Keep quarantined; separately construct and independently review an unambiguous question and answer.",
  [("before_action", "Martian Manhunter", "Created by writer Joseph Samachson"), ("before_action", "Ma'alefa'ak", "Created by writer John Ostrander")]),
 (135, "What was the nickname given", "retain_candidate", ["query_copies_visible_fundraiser_relation"],
  "The For Darfur article identifies Lilly Pulitzer as the 2007 Palm Beach fundraiser's partner. The query follows that person, and the retrieved biography supplies Queen of Prep.", None,
  [("before_action", "For Darfur", "in conjunction with fashion designer Lilly Pulitzer"), ("retrieved_after_search", "Lilly Pulitzer", "called the \"Queen of Prep\"")]),
 (136, "What was the nickname given", "retain_candidate", [],
  "Lilly Pulitzer is named in the fundraiser report, and her biography explicitly gives the nickname Queen of Prep. The answer is supported.", None,
  [("before_action", "For Darfur", "in conjunction with fashion designer Lilly Pulitzer"), ("before_action", "Lilly Pulitzer", "called the \"Queen of Prep\"")]),
 (137, "What rock band had a song featured", "quarantine_pending_repair", ["album_title_used_as_band_answer"],
  "The Glitch snippet identifies Tony Ayres as executive producer. Retrieval reaches Bonk, an album, whose article says the band is Big Pig and its song Breakaway appeared in Glitch. The frozen paired answer Bonk is the album title, not the requested rock band.",
  "Exclude the frozen pair; a new item could answer Big Pig after explicitly checking the album-to-band-to-song chain and independent review.",
  [("before_action", "Glitch (TV series)", "Tony Ayres serving as an Executive Producer"), ("retrieved_after_search", "Bonk (album)", "album by Australian rock band Big Pig"), ("retrieved_after_search", "Bonk (album)", "\"Breakaway\" was also featured in Season 1, Episode 4 of \"Glitch")]),
 (138, "What rock band had a song featured", "quarantine_pending_repair", ["answer_type_album_instead_of_band"],
  "Bonk is the album title. Its article calls Big Pig the rock band and says Breakaway appeared in Glitch; Glitch's article names Tony Ayres as executive producer. The answer fails the requested entity type.",
  "Keep the frozen answer quarantined; a corrected Big Pig answer needs new bound input and independent review.",
  [("before_action", "Glitch (TV series)", "Tony Ayres serving as an Executive Producer"), ("before_action", "Bonk (album)", "album by Australian rock band Big Pig"), ("before_action", "Bonk (album)", "\"Breakaway\" was also featured in Season 1, Episode 4 of \"Glitch")]),
 (139, "Rick Heinrichs performed design work", "repair_candidate", ["specific_design_work_on_film_not_explicitly_supported"],
  "Rick Heinrichs's snippet says he is a production designer and is well known for work on The Nightmare Before Christmas; it does not specify the nature of his work on that film. The search retrieves the film and Henry Selick as director. The question's 'performed design work on' overstates the visible relation.",
  "Ask which director made the stop-motion dark fantasy film on which Rick Heinrichs worked; replay the pair.",
  [("before_action", "Rick Heinrichs", "well known for his work on the \"Pirates of the Caribbean\" series, Ang Lee's \"Hulk\" and \"The Nightmare Before Christmas\""), ("retrieved_after_search", "The Nightmare Before Christmas", "directed by Henry Selick")]),
 (140, "Rick Heinrichs performed design work", "repair_candidate", ["paired_overstated_design_role"],
  "Henry Selick is explicitly named as director. The source connects Heinrichs with work on the film but does not establish that it was design work specifically; the wording must be weakened to the supported relation.",
  "Retain Henry Selick with a revised 'worked on' question after replay and independent review.",
  [("before_action", "Rick Heinrichs", "well known for his work on the \"Pirates of the Caribbean\" series, Ang Lee's \"Hulk\" and \"The Nightmare Before Christmas\""), ("before_action", "The Nightmare Before Christmas", "directed by Henry Selick")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-059.jsonl", cohort="v122_internal_candidates")
