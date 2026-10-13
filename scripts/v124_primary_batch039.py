"""Primary examination of exposed historical internal states 31–40."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (31, "Name Eirik Blodøks's", "retain_candidate", ["query_repeats_visible_title", "exposed_development_set"],
  "The visible Eric Bloodaxe article equates that name with Eirik Blodøks but omits his wife. Repeating the title retrieves a separate article that explicitly names Gunnhild as his wife.", None,
  [("before_action", "Eric Bloodaxe", "Norwegian: \"Eirik Blodøks\""), ("retrieved_after_search", "Ragnhild Eriksdotter", "Eric Bloodaxe and his wife, Gunnhild")]),
 (32, "Name Eirik Blodøks's", "retain_candidate", ["exposed_development_set"],
  "The first document resolves Eirik Blodøks to Eric Bloodaxe, and Ragnhild's biography identifies Gunnhild as Eric's wife. The answer is supported.", None,
  [("before_action", "Eric Bloodaxe", "Norwegian: \"Eirik Blodøks\""), ("before_action", "Ragnhild Eriksdotter", "Eric Bloodaxe and his wife, Gunnhild")]),
 (33, "The Shadow co-starred", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The film article lists multiple co-stars but does not identify which meets the character-actor description. The copied cast list retrieves Peter Boyle's biography, which supplies that qualifier.", None,
  [("before_action", "The Shadow (1994 film)", "Peter Boyle"), ("retrieved_after_search", "Peter Boyle", "Known as a character actor")]),
 (34, "The Shadow co-starred", "retain_candidate", ["exposed_development_set"],
  "The Shadow lists Peter Boyle among co-stars; his biography names him Peter Lawrence Boyle and describes him as an American character actor. The answer is supported.", None,
  [("before_action", "The Shadow (1994 film)", "Peter Boyle"), ("before_action", "Peter Boyle", "Peter Lawrence Boyle"), ("before_action", "Peter Boyle", "Known as a character actor")]),
 (35, "What is the nationality of the composer", "retain_candidate", ["query_repeats_visible_title", "exposed_development_set"],
  "Mignon's article identifies Franz Schubert as the composer but gives no nationality. Repeating the document title retrieves Schubert's biography, which says Austrian.", None,
  [("before_action", "Mignon (Schubert)", "several lieder by Franz Schubert"), ("retrieved_after_search", "Franz Schubert", "an Austrian composer")]),
 (36, "What is the nationality of the composer", "retain_candidate", ["exposed_development_set"],
  "The Mignon article attributes the lieder to Franz Schubert and his biography identifies him as Austrian. The answer is supported.", None,
  [("before_action", "Mignon (Schubert)", "several lieder by Franz Schubert"), ("before_action", "Franz Schubert", "an Austrian composer")]),
 (37, "What year was the actor", "repair_candidate", ["false_actor_premise", "query_copies_visible_context", "exposed_development_set"],
  "The source says the new pilot starred Sam Cooke but does not call him an actor; his retrieved biography calls him a singer, songwriter, and entrepreneur. The birth year is retrievable, but the question's actor premise needs correction to singer or performer.", None,
  [("before_action", "Shindig!", "new pilot with a new cast of artists was shot starring Sam Cooke"), ("retrieved_after_search", "Sam Cooke", "an American singer, songwriter, and entrepreneur")]),
 (38, "What year was the actor", "repair_candidate", ["false_actor_premise", "exposed_development_set"],
  "The 1931 birth year belongs to Sam Cooke, whom the retrieved article describes as a singer rather than an actor. The answer year is visible, but the question mischaracterizes the target and should be repaired as a pair.", None,
  [("before_action", "Shindig!", "starring Sam Cooke"), ("before_action", "Sam Cooke", "January 22, 1931"), ("before_action", "Sam Cooke", "an American singer, songwriter, and entrepreneur")]),
 (39, "In what city was the band", "retain_candidate", ["query_copies_distractor_context", "exposed_development_set"],
  "The box-set article already identifies ZZ Top as its band. The query copies a song article instead of targeting the band; nevertheless the returned ZZ Top article gives the missing formation city. Query quality is weak despite a supported result.", None,
  [("before_action", "Chrome, Smoke &amp; BBQ", "box set by American blues-rock band ZZ Top"), ("retrieved_after_search", "ZZ Top", "formed in 1969 in Houston, Texas")]),
 (40, "In what city was the band", "retain_candidate", ["exposed_development_set"],
  "Chrome, Smoke & BBQ is attributed to ZZ Top, and the band article states that it formed in Houston, Texas. The answer is supported.", None,
  [("before_action", "Chrome, Smoke &amp; BBQ", "box set by American blues-rock band ZZ Top"), ("before_action", "ZZ Top", "formed in 1969 in Houston, Texas")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-039.jsonl", cohort="v117_exposed_internal50")
