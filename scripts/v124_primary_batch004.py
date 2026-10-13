"""Primary case examination of original training rows 31–40."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (31,"Which shopping center features", "retain_candidate", ["nonexclusive_satisfying_answer"],
  "Eddie Bauer is explicitly headquartered in Bellevue and Friendly Center explicitly lists it as a retailer. This supports a satisfying shopping-center answer within the evidence. It does not imply that Friendly Center is the only such shopping center in the world.", None,
  [("before_action","Eddie Bauer","headquartered in Bellevue, Washington"),("before_action","Friendly Center","The GAP, Eddie Bauer, Talbots")]),
 (32,"How many bracelet events", "retain_candidate", [],
  "The 44th WSOP is explicitly identified as the 2013 event, with 62 bracelet events at Rio. The Rio article verifies Caesars ownership and operation. The 40th WSOP distractor's 57 must not replace the requested 44th event count.", None,
  [("before_action","2013 World Series of Poker","The 2013 World Series of Poker was the 44th annual World Series of Poker"),("before_action","2013 World Series of Poker","There was 62 bracelet events"),("before_action","Rio All Suite Hotel and Casino","It is owned and operated by Caesars Entertainment Corporation.")]),
 (33,"What is the name of this industrial city", "retain_candidate", [],
  "The goalkeeper biography explicitly gives Tauragė as birthplace; the city article establishes that it is an industrial city in Lithuania. The geographically different Navahrudak distractor is irrelevant.", None,
  [("before_action","Ernestas Šetkus","born 25 May 1985 in Tauragė"),("before_action","Tauragė","is an industrial city in Lithuania")]),
 (34,"Alessandro is an opera", "retain_candidate", [],
  "The opera article directly links Alessandro to Handel, and the composer biography calls him a baroque composer and describes Baroque influences. Lowercase baroque is an acceptable rendering of the requested style.", None,
  [("before_action","Alessandro (opera)","is an opera composed by George Frideric Handel in 1726"),("before_action","George Frideric Handel","was a German, later British, baroque composer")]),
 (35,"How is the film about a tabloid", "repair_candidate", ["nonessential_unanchored_current"],
  "The prior biography identifies the 1991-debut talk show and host. A host-name lookup retrieves Ringmaster, which explicitly describes its show as similar to Springer's. The search is grounded and useful. However, 'current real life talk show' imposes an unanchored time claim unnecessary for the fixed debut-year identification.",
  "Remove only 'current' from the question, retaining the 1991 debut condition. Rebuild all same-question states with the revised prompt and verify the later answer separately; the current target remains a search, not a newly invented similarity answer.",
  [("before_action","Jerry Springer","He has hosted the tabloid talk show \"Jerry Springer\" since its debut in 1991"),("retrieved_after_search","Ringmaster (film)","host of a show similar to his own, in this case called simply \"Jerry\".")]),
 (36,"Taylor McLam composed music", "retain_candidate", [],
  "McLam's advertising-client list includes McDonald's and the company article explicitly dates its founding to 1940. The brand relation and year are both supported; later franchise and acquisition dates do not change the requested founding answer.", None,
  [("before_action","Taylor McLam","Sears, Lego, McDonald's and Coca-Cola."),("before_action","McDonald's","It was founded in 1940 as a barbecue restaurant")]),
 (37,"What is the birth name of the singer", "repair_candidate", ["unsupported_inspiration_premise"],
  "The Lana Del Rey biography supports Elizabeth Woolridge Grant as the person's name. The Mintz article reports an Entertainment Weekly comparison of her musical sound with Lana Del Rey, not that Del Rey inspired Mintz. The name target is supported but the question's inspiration claim exceeds the evidence.",
  "Proposed question: What is the real name of the singer known as Lana Del Rey, whose music Entertainment Weekly used in describing Julie Mintz's sound? Preserve the existing name target only as a draft, replay the changed input and check all paired states before independent review.",
  [("before_action","Julie Mintz","calls Mintz's music \"an elegant and bewitching blend of Americana and orchestral pop that sits somewhere between Gillian Welch and Lana Del Rey.\""),("before_action","Lana Del Rey","Elizabeth Woolridge Grant (born June 21, 1985), known professionally as Lana Del Rey")]),
 (38,"What British female pop music", "retain_candidate", ["documented_song_title_variants"],
  "The Buffalo G article explicitly identifies Bananarama's 1982 cover and the Velvelettes original; the Bananarama article establishes the group's British/female/pop descriptors. The song-title variations are expressly documented, so they do not require an invented exact-title rejection.", None,
  [("before_action","Buffalo G","a rap cover version of the 1982 song by Bananarama, itself a cover of the 1964 original version by the Velvelettes."),("before_action","Bananarama","Bananarama are a British female pop music vocal group")]),
 (39,"Which Italian composer's music", "retain_candidate", ["broad_topic_query", "query_specificity_not_optimality_proof"],
  "Prior evidence lists Monteverdi among composers performed and characterizes Alessandrini as an interpreter of early Italian opera, but lacks which composer spans Renaissance and Baroque. The copied topical sentence is broad and pronoun-led; nevertheless it is grounded in the prior text and retrieves a listed composer with the missing period relation. This establishes useful retrieval, not that the query is optimal. Do not turn query style into a false semantic denial.", None,
  [("before_action","Rinaldo Alessandrini","performing music of Monteverdi, Vivaldi, Couperin, Bach, and others."),("before_action","Rinaldo Alessandrini","He is considered a foremost interpreter of early Italian opera."),("retrieved_after_search","Claudio Monteverdi","a crucial transitional figure between the Renaissance and the Baroque periods of music history.")]),
 (40,"Where is the investment holding", "retain_candidate", [],
  "Milwaukee is explicitly a brand of Techtronic Industries and the latter article identifies the investment holding company as based in Hong Kong. This ordinary ownership/location join does not require both clauses in one document.", None,
  [("before_action","Milwaukee Electric Tool Corporation","The Milwaukee Electric Tool Corporation is a brand of Techtronic Industries"),("before_action","Techtronic Industries","is an investment holding company based in Hong Kong.")]),
]

if __name__ == "__main__":
 save_batch(SPECS,"primary-batch-004.jsonl")
