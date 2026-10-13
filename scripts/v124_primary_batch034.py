"""Primary case examination of original training rows 331–340."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (331, "Name the 2005 romantic comedy", "retain_candidate", ["query_known_film_verifies_writer", "duplicate_training_action_group"],
  "The Wedding Date's year and Asking for Trouble adaptation are visible before search. Querying the candidate film retrieves Dana Fox's biography naming her as writer of both The Wedding Date and What Happens in Vegas, supplying the link missing from row 330's retrieval.", None,
  [("before_action", "The Wedding Date", "Based on the novel \"Asking for Trouble\" by Elizabeth Young"), ("retrieved_after_search", "Dana Fox", "writer of \"The Wedding Date\" (2005), \"What Happens in Vegas\" (2008)")]),
 (332, "Name the 2005 romantic comedy", "repair_candidate", ["duplicate_training_action"],
  "Same question, candidate-film query and Dana Fox cross-film writing credit as row 331. It is redundant supervision.",
  "Keep original state and deduplicate after paired answer inspection.",
  [("before_action", "The Wedding Date", "Based on the novel \"Asking for Trouble\" by Elizabeth Young"), ("retrieved_after_search", "Dana Fox", "writer of \"The Wedding Date\" (2005), \"What Happens in Vegas\" (2008)")]),
 (333, "What profession does Izu Ojukwu", "retain_candidate", [],
  "Ojukwu's biography calls him a film director, and Savage's calls him a television director. Director is the shared profession, despite their different mediums.", None,
  [("before_action", "Izu Ojukwu", "a Nigerian film director"), ("before_action", "Fred Savage", "television director")]),
 (334, "The winner of the 1976 United States Senate", "retain_candidate", [],
  "The 1976 Missouri Senate election article names John Danforth as winner, and his biography explicitly says he is an ordained Episcopal priest.", None,
  [("before_action", "United States Senate election in Missouri, 1976", "Republican John Danforth won the open seat"), ("before_action", "John Danforth", "an ordained Episcopal priest")]),
 (335, "What star of \"Down to Earth\"", "retain_candidate", ["film_title_disambiguation"],
  "Marc Platt's pre-action biography places him in the 1954 Seven Brides for Seven Brothers film; the search retrieves the 1947 Down to Earth film naming Marc Platt as star. This distinguishes the film from other works sharing its title.", None,
  [("before_action", "Marc Platt (dancer)", "one of the seven brothers in the film \"Seven Brides for Seven Brothers\""), ("retrieved_after_search", "Down to Earth (1947 film)", "starring Rita Hayworth, Larry Parks, and Marc Platt")]),
 (336, "Where does the college which John Abercrombie", "retain_candidate", ["query_copies_visible_context"],
  "Abercrombie's biography identifies Berklee College of Music; the copied-sentence query retrieves Berklee's article saying it also operates a campus in Valencia, Spain. The location answer is directly supported.", None,
  [("before_action", "John Abercrombie (guitarist)", "studied at Berklee College of Music in Boston, Massachusetts"), ("retrieved_after_search", "Berklee College of Music", "also operated a campus in Valencia, Spain")]),
 (337, "Dawn Penn", "retain_candidate", ["query_known_singer_verifies_album"],
  "The singer article identifies Dawn Penn and her hit but not its album year. Querying her name retrieves the debut album No, No, No and its 1994 release; the song article also states the hit is from that album.", None,
  [("before_action", "Dawn Penn", "single \"You Don't Love Me (No, No, No)\""), ("retrieved_after_search", "No, No, No (Dawn Penn album)", "debut album by a Jamaican reggae singer Dawn Penn, released in 1994"), ("retrieved_after_search", "You Don't Love Me (No, No, No)", "from her debut studio album, \"No, No, No\" (1994)")]),
 (338, "Who did the box cover art", "repair_candidate", ["pen_name_authorship_ambiguous"],
  "Sargent's biography says Keith Martin was a pen name for Fighting Fantasy gamebooks, while Night Below credits Carl Sargent as writer and Jeff Easley as box artist. The source does not say Night Below was made under the Keith Martin name, repeating row 186's muddled question premise.",
  "Apply the existing question-level draft naming Night Below and Carl Sargent directly, then replay this search and answer row 186 for independent review.",
  [("before_action", "Carl Sargent", "using the pen name Keith Martin to write Fighting Fantasy gamebooks"), ("retrieved_after_search", "Night Below", "was written by Carl Sargent, with box cover art by Jeff Easley")]),
 (339, "What album is Khia's best known", "retain_candidate", ["query_known_artist_fills_album"],
  "Khia's biography identifies My Neck, My Back as her best-known hit; searching her name retrieves the song article naming the debut album Thug Misses.", None,
  [("before_action", "Khia", "best known for her hit single \"My Neck, My Back (Lick It)\""), ("retrieved_after_search", "My Neck, My Back (Lick It)", "lead single from her debut album, \"Thug Misses\"")]),
 (340, "What did Willis H. O'Brien", "retain_candidate", ["question_restated_in_query"],
  "O'Brien's biography lists King Kong among his effects work; the retrieved 1933 King Kong film article names Bruce Cabot and Robert Armstrong and its New York opening. The film is uniquely supported among the retrieved O'Brien works.", None,
  [("before_action", "Willis H. O'Brien", "work on \"The Lost World\" (1925), \"King Kong\" (1933) and \"Mighty Joe Young\""), ("retrieved_after_search", "King Kong (1933 film)", "It stars Fay Wray, Bruce Cabot and Robert Armstrong, and opened in New York City")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-034.jsonl")
