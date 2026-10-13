"""Primary case examination of original training rows 71–80."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (71, "Arthur Fletcher and Joe McCarthy", "retain_candidate", ["name_variant_art_arthur"],
  "The Art Fletcher biography names Arthur Fletcher and links him to the Yankees of Joe McCarthy as coach. The McCarthy manager biography independently names the New York Yankees. The unrelated musician result does not change this supported join.", None,
  [("before_action", "Art Fletcher", "Arthur Fletcher"), ("before_action", "Art Fletcher", "the Yankees of Miller Huggins and Joe McCarthy as a coach"), ("before_action", "Joe McCarthy (manager)", "teams of the New York Yankees")]),
 (72, "What major metropolitan area", "retain_candidate", [],
  "The press article identifies a publisher of popular reference works headquartered in Metro Detroit. The area article matches the geographic description. This is the answer counterpart of the earlier useful search at row 70.", None,
  [("before_action", "Visible Ink Press", "a publisher of popular reference works.  Its headquarters are in Canton Charter Township, Michigan in Metro Detroit"), ("before_action", "Metro Detroit", "a major metropolitan area in Southeast Michigan")]),
 (73, "What album is Khia", "retain_candidate", [],
  "Khia's article identifies her best-known single, and the single article names Thug Misses as its debut album. The answer is a direct two-document join.", None,
  [("before_action", "Khia", "best known for her hit single \"My Neck, My Back (Lick It)\""), ("before_action", "My Neck, My Back (Lick It)", "lead single from her debut album, \"Thug Misses\"")]),
 (74, "\"The Simpsons\"' fourteenth", "retain_candidate", [],
  "The season article singles out How I Spent My Strummer Vacation as the Mike Scully-run episode. Its episode article gives the US first-air date November 10, 2002. The Parks and Recreation result is merely a same-name Scully distractor.", None,
  [("before_action", "The Simpsons (season 14)", "The other episode, \"How I Spent My Strummer Vacation\", was run by Mike Scully."), ("before_action", "How I Spent My Strummer Vacation", "It first aired on the Fox network in the United States on November 10, 2002.")]),
 (75, "What Walt Disney Feature Animation", "retain_candidate", [],
  "Bettin's biography links his Dawson voice role to The Great Mouse Detective. The film article verifies the Walt Disney Feature Animation production condition. The Val Kilmer and Val Lewton results are unrelated.", None,
  [("before_action", "Val Bettin", "voicing Dr. David Q. Dawson in the 1986 Disney animated film \"The Great Mouse Detective\""), ("before_action", "The Great Mouse Detective", "produced by Walt Disney Feature Animation")]),
 (76, "What was the profession", "retain_candidate", ["ellery_queen_pseudonym_and_character"],
  "The prior Ellery Queen article distinguishes the writing duo's pseudonym from its fictional writer character. The search explicitly names both compared entities and retrieves John Fowles as a novelist. 'Writer' is a supported shared profession, provided the answer state keeps the pseudonym/character distinction clear.", None,
  [("before_action", "Ellery Queen", "wrote, edited, and anthologized detective fiction under the pseudonym of Ellery Queen"), ("retrieved_after_search", "John Fowles", "was an English novelist")]),
 (77, "In how many countries", "retain_candidate", ["time_varying_availability_snapshot"],
  "Jedd Hughes's birthplace is Quorn, Australia; the namesake meat substitute's article states availability in 19 countries. The answer matches that source snapshot, though availability is time varying and should not be used as a present-day fact.", None,
  [("before_action", "Jedd Hughes", "born in Quorn, Australia"), ("before_action", "Quorn", "available in 19 countries")]),
 (78, "In which state", "retain_candidate", [],
  "The team article places home games at West Memorial Field on Manchester High School West's campus. The school article locates that campus in Manchester, New Hampshire, and the team article also supplies the state.", None,
  [("before_action", "Manchester Freedom", "Home games are played at West Memorial Field on the campus of Manchester High School West."), ("before_action", "Manchester High School West", "located in Manchester, New Hampshire")]),
 (79, "How many homicides", "retain_candidate", ["query_verifies_investigator_relation"],
  "Bundy's biography already says he confessed to 30 homicides. The search for the visible killer name retrieves Keppel's biography and verifies that Keppel investigated Bundy; it thus fills a relation even though the numeric answer was already visible.", None,
  [("before_action", "Ted Bundy", "he confessed to 30 homicides"), ("retrieved_after_search", "Robert D. Keppel", "his work investigating serial killers Ted Bundy and Gary Ridgway")]),
 (80, "In which air combat", "retain_candidate", [],
  "The airport article identifies Joe Foss as its honoree but lacks the battle. The grounded query repeats that name relation and retrieves Foss's biography, which explicitly ties his Medal of Honor to air combat in the Guadalcanal Campaign.", None,
  [("before_action", "Sioux Falls Regional Airport", "Named in honor of aviator Joe Foss"), ("retrieved_after_search", "Joe Foss", "Medal of Honor in recognition of his role in air combat during the Guadalcanal Campaign")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-008.jsonl")
