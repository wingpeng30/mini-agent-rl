"""Primary examination of v123 internal candidate states 51–60."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (51, "Marvin Camel was born in what Montana county", "retain_candidate", ["query_uses_visible_birthtown", "retrieval_adds_county"],
  "The initial Marvin Camel article says he was born in Ronan and separately fought out of Missoula. The query preserves both, and the returned Ronan article explicitly places that birth town in Lake County; the Missoula County distractor should not be used.", None,
  [("before_action", "Marvin Camel", "He was born in Ronan and fought out of Missoula"), ("retrieved_after_search", "Ronan, Montana", "city in Lake County, Montana")]),
 (52, "Marvin Camel was born in what Montana county", "retain_candidate", [],
  "Ronan is Marvin Camel's stated birth town, and the Ronan article places it in Lake County. The answer does not confuse his Missoula fighting base with birthplace.", None,
  [("before_action", "Marvin Camel", "born in Ronan and fought out of Missoula"), ("before_action", "Ronan, Montana", "city in Lake County, Montana")]),
 (53, '"Burn" and "Walk Away" were featured', "repair_candidate", ["query_uses_unrelated_shirley_fire", "retrieval_incidentally_finds_film"],
  "The initial Resonance album article already identifies Reign of Fire as the film featuring Burn and Walk Away. The query instead searches Shirley Fire, a wildfire distractor. It happens to retrieve the Reign of Fire page and director Rob Bowman, but that does not ground the search action.",
  "Search the already visible film title with its director relation, then replay.",
  [("before_action", "Resonance (Mad at Gravity album)", "end title credits of the 2002 action film Reign of Fire"), ("before_action", "Shirley Fire", "wildfire"), ("retrieved_after_search", "Reign of Fire (film)", "directed by Rob Bowman")]),
 (54, '"Burn" and "Walk Away" were featured', "repair_candidate", ["paired_unrelated_wildfire_query", "answer_supported"],
  "The album-to-film link and the retrieved director credit support Rob Bowman. The paired action used an unrelated wildfire query, so the answer cannot rehabilitate that search example.",
  "Keep the director answer after a Reign of Fire targeted retrieval replay.",
  [("before_action", "Resonance (Mad at Gravity album)", "end title credits of the 2002 action film Reign of Fire"), ("before_action", "Reign of Fire (film)", "directed by Rob Bowman")]),
 (55, "According to the 2010 census, what was the population", "retain_candidate", ["query_uses_visible_miller_place_birth", "retrieval_adds_dated_population"],
  "Captain Robert Robinson's initial excerpt says he was born in Miller Place, New York. Searching that visible birthplace retrieves the Miller Place CDP article and its explicitly dated 2010-census population of 12,339.", None,
  [("before_action", "Captain Robert Robinson", "born in Miller Place, New York"), ("retrieved_after_search", "Miller Place, New York", "population was 12,339 at the 2010 census")]),
 (56, "According to the 2010 census, what was the population", "retain_candidate", [],
  "The birthplace relation points to Miller Place, and its census article states 12,339 in 2010. The question includes the date, avoiding an unanchored population snapshot.", None,
  [("before_action", "Captain Robert Robinson", "born in Miller Place, New York"), ("before_action", "Miller Place, New York", "population was 12,339 at the 2010 census")]),
 (57, "When was the inventor of the wobble board born", "repair_candidate", ["query_uses_visible_inventor", "paired_answer_fragment_malformed"],
  "The initial wobble-board article names Rolf Harris as inventor, and the query reaches his biography with the full 30 March 1930 date. The paired answer is malformed with a leading '(born', so the pair cannot be exported unchanged.",
  "Use the clean date only in a new answer state, then replay and independently audit the pair.",
  [("before_action", "Wobble board", "invented and popularized by the Australian musician and artist Rolf Harris"), ("retrieved_after_search", "Rolf Harris", "born 30 March 1930")]),
 (58, "When was the inventor of the wobble board born", "repair_candidate", ["answer_contains_parenthesis_and_born_fragment"],
  "The source gives 30 March 1930 as Rolf Harris's birth date. The frozen target '(born 30 March 1930' is an extracted sentence fragment, not a clean date answer.",
  "Create a hash-bound clean-date answer draft, replay its action-time input, and independently review it; do not overwrite the old target.",
  [("before_action", "Wobble board", "invented and popularized by the Australian musician and artist Rolf Harris"), ("before_action", "Rolf Harris", "born 30 March 1930")]),
 (59, "Jon B. Kutler serves as the Chairman of a subcommittee", "quarantine_pending_repair", ["jpl_location_misattributed_to_subcommittee", "no_subcommittee_site_evidence"],
  "The Kutler excerpt says he chairs the Jet Propulsion Laboratory Subcommittee of a Caltech trusteeship. The retrieved JPL article locates the laboratory in La Cañada Flintridge and Pasadena; it says nothing about where the subcommittee itself sits. Transferring the lab's physical location to a governance subcommittee is unsupported.",
  "If the intended question is about JPL's location, explicitly ask for the laboratory associated with his subcommittee and replay; otherwise obtain direct subcommittee meeting/location evidence.",
  [("before_action", "Jon B. Kutler", "Chairman of the Jet Propulsion Laboratory Subcommittee"), ("retrieved_after_search", "Jet Propulsion Laboratory", "NASA field center in La Cañada Flintridge, California and Pasadena, California")]),
 (60, "Jon B. Kutler serves as the Chairman of a subcommittee", "quarantine_pending_repair", ["paired_institution_location_conflation"],
  "The two California cities are JPL's location, not an established location of its trustees' subcommittee. The exact answer string is present, but it answers a different question.",
  "Do not admit the original pair; rewrite to the laboratory-location relation and revalidate only if the source task allows it.",
  [("before_action", "Jon B. Kutler", "Chairman of the Jet Propulsion Laboratory Subcommittee"), ("before_action", "Jet Propulsion Laboratory", "La Cañada Flintridge, California and Pasadena, California")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-106.jsonl", cohort="v123_internal_candidates")
