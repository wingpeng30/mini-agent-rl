"""Primary case examination of original training rows 151–160."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (151, "What is Jayson Anthony Paul", "retain_candidate", [],
  "Cryme Tyme's article names JTG as a member but does not identify his civilian name. The question-grounded query retrieves the JTG article linking Jayson Anthony Paul to the ring name. This pairs with supported answer row 88.", None,
  [("before_action", "Cryme Tyme", "consisting of JTG and Shad Gaspard"), ("retrieved_after_search", "JTG", "Jayson Anthony Paul (born December 10, 1984), better known by the ring name JTG")]),
 (152, "Who married the twin", "retain_candidate", [],
  "Hamnet's article identifies Judith as his twin and says he died at 11. Querying the visible Hamnet name retrieves Judith's biography, which explicitly names Thomas Quiney as her spouse. No answer is inferred from the unrelated twin-word distractors.", None,
  [("before_action", "Hamnet Shakespeare", "the fraternal twin of Judith Shakespeare.  He died at age 11"), ("retrieved_after_search", "Judith Quiney", "She married Thomas Quiney")]),
 (153, "What was the original name", "retain_candidate", ["organisation_historical_name_sequence"],
  "The Raging Spirits article identifies Walt Disney Imagineering as creator. The copied visible creation sentence retrieves the Imagineering article, which says its first name was Walt Disney, Inc. and later WED Enterprises. The first name is distinct from the later, better-known WED name.", None,
  [("before_action", "Raging Spirits", "Created by Walt Disney Imagineering"), ("retrieved_after_search", "Walt Disney Imagineering", "originally known as Walt Disney, Inc. then WED Enterprises")]),
 (154, "What were both the Locarno", "retain_candidate", [],
  "The Stresa Front is explicitly called an agreement and the Locarno Treaties are seven agreements. 'Agreements' is the supported common category.", None,
  [("before_action", "Stresa Front", "was an agreement made in Stresa"), ("before_action", "Locarno Treaties", "were seven agreements negotiated at Locarno")]),
 (155, "Which of the film", "retain_candidate", ["query_verifies_candidate_film_attributes"],
  "Handel's biography already lists Noah and The Fountain, so the answer candidate is visible but the 2014 biblical-drama attributes are missing. Searching his visible name retrieves Noah's film article with exactly those attributes and the co-writing relation. The unrelated composer Handel is a name-collision distractor.", None,
  [("before_action", "Ari Handel", "co-writing the films \"Noah\" and \"The Fountain\""), ("retrieved_after_search", "Noah (2014 film)", "a 2014 American epic biblical drama film directed by Darren Aronofsky")]),
 (156, "Yi Jing-Qian", "retain_candidate", [],
  "Yi's biography names the 2000 Australian Open as her third-round Grand Slam best. The Australian Open article places that tournament in Melbourne. The answer is supported.", None,
  [("before_action", "Yi Jing-Qian", "reached the 3rd round of the Australian Open in 2000"), ("before_action", "Australian Open", "held annually over the last fortnight of January in Melbourne, Australia")]),
 (157, "How many people lived", "retain_candidate", ["city_municipality_population_distinction"],
  "Tapia's biography says he was raised in Culiacán. The city article clearly distinguishes 675,773 city residents at the 2010 census from 858,638 in the municipality; the stored answer uses the city count asked for.", None,
  [("before_action", "Roberto Tapia", "raised in Culiacán, Sinaloa, Mexico"), ("before_action", "Culiacán", "675,773 inhabitants in the city (census of 2010) and 858,638 in the municipality")]),
 (158, "Vale of Tears", "retain_candidate", ["query_copies_partial_author_sentence"],
  "The novel article identifies Peter T. King as author and politician but omits his birth date. Its copied partial name/role sentence retrieves King's biography giving 1944. This fills the missing date, although the query is less direct than a name-and-birth-date lookup.", None,
  [("before_action", "Vale of Tears (novel)", "the third novel by Peter T. King, a member of the Republican Party"), ("retrieved_after_search", "Peter T. King", "Peter Thomas King (born April 5, 1944)")]),
 (159, "In how many countries", "retain_candidate", ["time_varying_availability_snapshot"],
  "The prior Hughes article gives his birthplace as Quorn but lacks the meat-product coverage. The copied birthplace sentence retrieves the Quorn product article saying 19 countries, thus resolving the namesake link. Its availability count is source-snapshot data rather than a current claim.", None,
  [("before_action", "Jedd Hughes", "born in Quorn, Australia"), ("retrieved_after_search", "Quorn", "available in 19 countries")]),
 (160, "What is the nationality", "retain_candidate", [],
  "The stadium article names Sepp Blatter as the FIFA president at its 2011 ribbon cutting. The copied ceremony sentence retrieves his biography, which identifies him as Swiss. The question refers to the historical ceremony, so the later presidency status is not used as a present-day claim.", None,
  [("before_action", "Nicaragua National Football Stadium", "attended by FIFA president Sepp Blatter"), ("retrieved_after_search", "Sepp Blatter", "is a Swiss football administrator")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-016.jsonl")
