"""Primary examination of v122 internal candidate states 101–110."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (101, "In what year was Mark Michael Brzezicki", "repair_candidate", ["most_well_known_subjective_vs_primarily_known"],
  "The initial biography says Brzezicki is primarily known for his work with Big Country, which the search reaches and dates to 1981. This supports the intended band, but 'most well known band' turns a statement about his work into an unqualified ranking of the bands he joined.",
  "Ask when Big Country, the band for which his work is primarily known, was formed; replay before approving the pair.",
  [("before_action", "Mark Brzezicki", "primarily known for his work with Big Country"), ("retrieved_after_search", "Big Country", "formed in Dunfermline, Fife, in 1981")]),
 (102, "In what year was Mark Michael Brzezicki", "repair_candidate", ["paired_subjective_superlative"],
  "The year 1981 is directly supported for Big Country, and the musician source links him to it. The original 'most well known' wording is stronger than the source's 'primarily known for his work' and should be repaired before this paired item is used.",
  "Retain the year only in a reworded, replayed pair naming the documented relation.",
  [("before_action", "Mark Brzezicki", "primarily known for his work with Big Country"), ("before_action", "Big Country", "in 1981")]),
 (103, '"Grey" is the fourth studio album', "retain_candidate", ["query_copies_visible_album_relation"],
  "The initial Grey album article names Sandy Lam as its singer. The search repeats that grounded relation and retrieves Lam's birth year, the missing fact.", None,
  [("before_action", "Grey (album)", "fourth studio album by Sandy Lam"), ("retrieved_after_search", "Sandy Lam", "born 26 April 1966")]),
 (104, '"Grey" is the fourth studio album', "retain_candidate", [],
  "Grey is identified as Sandy Lam's fourth studio album, and her biography says she was born in 1966. The answer is supported by the paired documents.", None,
  [("before_action", "Grey (album)", "fourth studio album by Sandy Lam"), ("before_action", "Sandy Lam", "born 26 April 1966")]),
 (105, "What was the debut film of the female star", "retain_candidate", ["query_copies_visible_cast_relation"],
  "Off the Rack names Ed Asner and Eileen Brennan as stars. Searching that grounded cast relation retrieves Brennan's biography with her film debut. The query does not contain the answer.", None,
  [("before_action", "Off the Rack", "stars Ed Asner and Eileen Brennan"), ("retrieved_after_search", "Eileen Brennan", "film debut in the satire \"Divorce American Style\"")]),
 (106, "What was the debut film of the female star", "retain_candidate", [],
  "The female star Eileen Brennan is named by the sitcom article, and her biography explicitly identifies Divorce American Style as her film debut.", None,
  [("before_action", "Off the Rack", "stars Ed Asner and Eileen Brennan"), ("before_action", "Eileen Brennan", "film debut in the satire \"Divorce American Style\"")]),
 (107, "Where did the bomber flown by Frederick Bock", "quarantine_pending_repair", ["false_premise_bock_did_not_pilot_bomb_dropping_aircraft", "answer_location_already_visible"],
  "The initial Bock article explicitly distinguishes his measurement plane The Great Artiste from bomb-dropping Bockscar, which was piloted on this raid by Charles Sweeney. The question wrongly makes Bock's flown bomber the one that dropped the bomb. Nagasaki was visible before search but cannot cure the false premise.",
  "Exclude the frozen pair. A new item must accurately distinguish Bock, Sweeney and the two aircraft, then be independently reviewed.",
  [("before_action", "Frederick C. Bock", "flying the B-29 bomber \"The Great Artiste\", which was used for scientific measurements"), ("before_action", "Frederick C. Bock", "Major Charles Sweeney piloted \"Bockscar\""), ("retrieved_after_search", "Charles Sweeney", "pilot who flew Bockscar carrying the Fat Man atomic bomb")]),
 (108, "Where did the bomber flown by Frederick Bock", "quarantine_pending_repair", ["false_premise_bock_did_not_pilot_bomb_dropping_aircraft", "correct_answer_to_different_question"],
  "Nagasaki is the bombing location, but the aircraft Bock flew on the raid was used for measurements and did not drop the bomb. The answer would suit a different question about Bockscar or the raid, not this false-premise question.",
  "Keep quarantined; create and audit a new accurately worded pair rather than counting this as supported.",
  [("before_action", "Frederick C. Bock", "The Great Artiste\", which was used for scientific measurements"), ("before_action", "Frederick C. Bock", "Major Charles Sweeney piloted \"Bockscar\""), ("before_action", "Charles Sweeney", "to the Japanese city of Nagasaki")]),
 (109, 'What war does "Unity of Command"', "retain_candidate", ["query_copies_visible_campaign_relation"],
  "The game's initial snippet places it in the 1942/43 Stalingrad Campaign on the Eastern Front. The search brings in the Battle of Stalingrad article, which explicitly identifies World War II. This is a grounded missing-war link.", None,
  [("before_action", "Unity of Command (video game)", "1942/43 Stalingrad Campaign on the Eastern Front"), ("retrieved_after_search", "Battle of Stalingrad", "major battle of World War II")]),
 (110, 'What war does "Unity of Command"', "retain_candidate", [],
  "Unity of Command covers the Stalingrad campaign, and the retrieved battle article places Stalingrad in World War II. The answer follows this chain.", None,
  [("before_action", "Unity of Command (video game)", "1942/43 Stalingrad Campaign on the Eastern Front"), ("before_action", "Battle of Stalingrad", "major battle of World War II")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-056.jsonl", cohort="v122_internal_candidates")
