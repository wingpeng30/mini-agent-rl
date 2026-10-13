"""Primary examination of v123 internal candidate states 151–160."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (151, "The 2016 United States presidential election in Ohio", "retain_candidate", ["candidate_inferred_from_visible_vice_president", "retrieval_confirms_election_link"],
  "The initial context identifies Mike Pence as a vice president and former Indiana governor, though it does not link him to the Ohio 2016 election. The person query is a plausible hypothesis from the question and retrieves the election passage identifying Pence as Trump's running mate and Indiana governor. It should not be counted as retrieving the state for the first time.", None,
  [("before_action", "Mike Pence", "Governor of Indiana from 2013 to 2017"), ("retrieved_after_search", "United States presidential election in Ohio, 2016", "running mate Indiana Governor Mike Pence")]),
 (152, "The 2016 United States presidential election in Ohio", "retain_candidate", [],
  "The election article explicitly names Indiana Governor Mike Pence as a running mate, and the Pence article independently places his governorship in Indiana. The 2016 framing avoids the later-tense ambiguity of 'current'.", None,
  [("before_action", "United States presidential election in Ohio, 2016", "running mate Indiana Governor Mike Pence"), ("before_action", "Mike Pence", "Governor of Indiana from 2013 to 2017")]),
 (153, "In which state did Joan Horvath", "retain_candidate", ["query_uses_visible_workplace", "retrieval_adds_state"],
  "The initial Horvath biography says she worked at JPL for sixteen years; retrieval places JPL's field center in California. The query repeats the workplace sentence but targets a genuine missing location relation.", None,
  [("before_action", "Joan Horvath", "Jet Propulsion Laboratory for sixteen years"), ("retrieved_after_search", "Jet Propulsion Laboratory", "La Cañada Flintridge, California and Pasadena, California")]),
 (154, "In which state did Joan Horvath", "retain_candidate", [],
  "Horvath's sixteen-year employer is JPL, and the JPL article locates the field center in California. The inference stays within the same named workplace.", None,
  [("before_action", "Joan Horvath", "Jet Propulsion Laboratory for sixteen years"), ("before_action", "Jet Propulsion Laboratory", "La Cañada Flintridge, California and Pasadena, California")]),
 (155, "Are Marilyn Manson and American Music Club", "repair_candidate", ["query_unrelated_american_eats", "person_band_nationality_mismatch"],
  "American Music Club is identified as an American band before search. The query appends American Eats, an unrelated TV show, and retrieval returns both Marilyn Manson the American person and the American band. The question leaves unclear whether Marilyn Manson means the musician or group and loosely applies nationality to a band.",
  "Disambiguate the musician or band, ask about country of origin if comparing groups, and replay a relevant query.",
  [("before_action", "American Music Club", "American, San Francisco-based indie rock band"), ("before_action", "American Eats", "American reality television show"), ("retrieved_after_search", "Marilyn Manson", "American singer"), ("retrieved_after_search", "Marilyn Manson (band)", "American rock band")]),
 (156, "Are Marilyn Manson and American Music Club", "repair_candidate", ["paired_ambiguous_entity", "paired_unrelated_query"],
  "Yes has support whether Marilyn Manson denotes the American performer or American band, but that coincidence does not resolve the question's entity type. The paired query is contaminated by an unrelated show.",
  "Rebuild around explicitly named entity types and independently review the answer.",
  [("before_action", "American Music Club", "American, San Francisco-based indie rock band"), ("before_action", "Marilyn Manson", "American singer"), ("before_action", "Marilyn Manson (band)", "American rock band")]),
 (157, "The largest city in Baja California", "quarantine_pending_repair", ["city_origin_not_proved", "site_relation_substituted_for_began_as"],
  "The land-grant article says its site is in present-day Tijuana, and the Tijuana article says it is Baja California's largest city. Neither passage says that Tijuana began as that land grant. The question's historical origin relation is stronger than the geographic overlap shown.",
  "Locate direct historical founding/origin evidence or rewrite as a site-location question; do not retain the present answer as proof of 'began as'.",
  [("before_action", "Rancho Cueros de Venado", "site is in present-day Tijuana"), ("retrieved_after_search", "Tijuana", "largest city in Baja California")]),
 (158, "The largest city in Baja California", "quarantine_pending_repair", ["paired_origin_relation_unsupported"],
  "Rancho Cueros de Venado is documented at a site in present-day Tijuana. That does not establish that Tijuana itself began as the grant, so the answer lacks support for the original wording.",
  "Require a direct origin source or a fully re-reviewed location question.",
  [("before_action", "Rancho Cueros de Venado", "site is in present-day Tijuana"), ("before_action", "Tijuana", "largest city in Baja California")]),
 (159, "The only bird commonly found in North America", "quarantine_pending_repair", ["missing_mockingbird_qualifier", "state_not_unique"],
  "The source calls the northern mockingbird the only *mockingbird* commonly found in North America. The question drops 'mockingbird', making its premise false and its state underspecified. The retrieved mascot passage says a mockingbird is Tennessee's state bird, but the initial American robin article already lists three other states for another bird.",
  "Restore 'mockingbird' and verify whether the intended bird-state relation is unique before rebuilding.",
  [("before_action", "Northern mockingbird", "only mockingbird commonly found in North America"), ("before_action", "American robin", "state bird of Connecticut, Michigan, and Wisconsin"), ("retrieved_after_search", "Scrappy Moc", "state bird of Tennessee")]),
 (160, "The only bird commonly found in North America", "quarantine_pending_repair", ["paired_false_premise", "state_not_unique"],
  "Tennessee is supported for a mockingbird, but the question omits the category that made the bird identifiable. The raw answer cannot repair that false, broad premise.",
  "Use a qualified question and independently check the state relation.",
  [("before_action", "Northern mockingbird", "only mockingbird commonly found in North America"), ("before_action", "Scrappy Moc", "state bird of Tennessee")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-116.jsonl", cohort="v123_internal_candidates")
