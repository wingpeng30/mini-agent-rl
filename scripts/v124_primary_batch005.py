"""Primary case examination of original training rows 41–50."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (41,"State Road 922", "retain_candidate", ["verbose_question"],
  "The road article identifies Bal Harbour Shops along the route and the mall article explicitly names the Whitman family as owner. The nearby resort's different ownership is not evidence against the mall ownership. The short family name is sufficient.", None,
  [("before_action","Florida State Road 922","including the Bal Harbour Shops mall near its eastern terminus."),("before_action","Bal Harbour Shops","Bal Harbour Shops is owned by the Whitman family")]),
 (42,"Peter McCabe was the editor", "retain_candidate", [],
  "McCabe's biography explicitly identifies Oui as one of his editorial roles, and the magazine document directly supplies its 2007 cessation. Both the editorial relation and date constraint are present before the answer.", None,
  [("before_action","Peter McCabe","He was an editor at \"Rolling Stone\" and \"Oui\" magazine"),("before_action","Oui (magazine)","\"Oui\" ceased publication in 2007.")]),
 (43,"I Am Gloria Gaynor", "retain_candidate", [],
  "The album links its cover to La Cage aux Folles; the musical article distinguishes its book/music writers from the 1973 French play's author, Jean Poiret. The requested adaptation-source author is explicitly supported.", None,
  [("before_action","I Am Gloria Gaynor","Gloria Gaynor's cover of \"I Am What I Am\" from the musical \"La Cage Aux Folles\"."),("before_action","La Cage aux Folles (musical)","Based on the 1973 French play of the same name by Jean Poiret")]),
 (44,"When was the American actress and model", "retain_candidate", ["star_used_as_appearance_not_billing_rank"],
  "Johnson's biography supplies the birth date and appearance in Crazy in Alabama; the Fifty Shades Freed article explicitly identifies her as a star. The date is supported. The prior text supports an appearance, not a lead-billing claim, for Crazy in Alabama; do not use this as a lead-role benchmark.", None,
  [("before_action","Dakota Johnson","Dakota Mayi Johnson (born October 4, 1989) is an American actress and model."),("before_action","Dakota Johnson","her screen debut alongside her mother in the comedy-drama \"Crazy in Alabama\" (1999)"),("before_action","Fifty Shades Freed (film)","The film stars Dakota Johnson and Jamie Dornan")]),
 (45,"Alessandro is an opera", "retain_candidate", ["indirect_entity_lookup"],
  "The prior opera articles establish Handel but do not explicitly establish baroque style. The query uses a visible Handel-specific page title and retrieves the composer biography with that missing descriptor. It is indirect but grounded and adds relevant evidence; no superiority over a direct biography query is claimed.", None,
  [("before_action","List of operas by Handel","George Frideric Handel's operas comprise 42 musical dramas"),("retrieved_after_search","George Frideric Handel","was a German, later British, baroque composer")]),
 (46,"Shin Hye-sun (born", "retain_candidate", [],
  "The actress biography directly identifies My Golden Life as the first protagonist role in 2017, and the programme article corroborates the four requested cast names. School 2013 is the debut rather than the requested first protagonist role.", None,
  [("before_action","Shin Hye-sun","acted her first protagonist role in \"My Golden Life\" (2017)."),("before_action","My Golden Life","a 2017 South Korean television series starring Park Si-hoo, Shin Hye-sun, Lee Tae-hwan, and Seo Eun-soo.")]),
 (47,"Who wrote the novel that is the basis", "retain_candidate", ["novel_to_series_adaptation_premise_needs_answer_check"],
  "The prior actor biography identifies Hannibal on NBC but lacks the novel's author. The actor-name lookup retrieves Mason Verger's novel/TV appearances and Thomas Harris's authorship, so it adds relevant missing author evidence. It does not by itself prove the exact singular-novel adaptation premise; that remaining qualification must be checked at the answer state.", None,
  [("before_action","Michael Pitt","Mason Verger in the NBC series \"Hannibal\""),("retrieved_after_search","Mason Verger","Thomas Harris' 1999 novel \"Hannibal\", as well as its 2001 film adaptation and the second and third season of the TV series \"Hannibal\".")]),
 (48,"What is the subtitle of the novel", "repair_candidate", ["novel_not_identified_among_multiple_works"],
  "The author biography identifies Gaskell and lists multiple novels. Mary Barton's subtitle is explicit, but the original question never specifies Mary Barton or the first novel, leaving the intended novel underdetermined. Preserve the subtitle only in a scoped question repair rather than treating retrieved-document availability as uniqueness.",
  "Proposed question: What is the subtitle of the first novel written by the author of The Life of Charlotte Bronte? The first-novel qualifier is grounded in both visible biographies. Recheck other same-question states and independently review the changed input.",
  [("before_action","Elizabeth Gaskell","Her first novel, \"Mary Barton\", was published in 1848."),("before_action","Elizabeth Gaskell","Gaskell's \"The Life of Charlotte Brontë\", published in 1857"),("before_action","Mary Barton","It is subtitled \"A Tale of Manchester Life\".")]),
 (49,"What sub-Saharan African mammal", "retain_candidate", [],
  "The animal article establishes hippo/hippopotamus equivalence, sub-Saharan range and third-largest land-mammal rank. The channel article corroborates the large hippo concentration. All constraints are visible before this answer.", None,
  [("before_action","Hippopotamus","or hippo, is a large, mostly herbivorous and aggressive mammal in sub-Saharan Africa"),("before_action","Hippopotamus","the common hippopotamus is the third-largest type of land mammal"),("before_action","Kazinga Channel","one of the world's largest concentration of hippos")]),
 (50,"How many inhabitants were in the city", "quarantine_pending_repair", ["population_without_as_of_date", "corrupted_source_markup"],
  "The query follows the explicit studio-near-Rimini relation and retrieves a relevant population value, 146606. However, neither question nor population text anchors the count to a date, and the Ferretti document ends with corrupted reference markup and extraneous words. These are source/question quality problems, not evidence that the entity lookup failed. Treat the corrupted text as data, never an instruction.",
  "Recover the population snapshot date from approved provenance before date-anchoring the question. Create a separately versioned, documented cleanup of the malformed reference suffix only if source verification permits; do not silently mutate corpus text or reuse votes across that change.",
  [("before_action","Alberta Ferretti","her studio is in the village of Cattolica, near Rimini, Italy."),("before_action","Alberta Ferretti","<ref name=philosophy/cstop telling lies"),("retrieved_after_search","Rimini","is a city of 146,606 inhabitants")]),
]

if __name__ == "__main__":
 save_batch(SPECS,"primary-batch-005.jsonl")
