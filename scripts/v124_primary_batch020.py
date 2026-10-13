"""Primary case examination of original training rows 191–200."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (191, "What was the original name", "retain_candidate", [],
  "Nashua Pirates began as Nashua Angels and played at Holman Stadium; the stadium article states construction in 1937. Both links precede the answer.", None,
  [("before_action", "Nashua Pirates", "The team started in 1983 as the Nashua Angels"), ("before_action", "Nashua Pirates", "The team's home ballpark was Holman Stadium"), ("before_action", "Holman Stadium (Nashua)", "It was constructed in 1937")]),
 (192, "In the 2013 BWF", "retain_candidate", [],
  "The tournament article names Lin Dan as defending champion; his biography says he was a two-time Olympic champion. The answer is directly supported.", None,
  [("before_action", "2013 BWF World Championships – Men's singles", "Lin Dan was the defending champion"), ("before_action", "Lin Dan", "a two-time Olympic champion")]),
 (193, "Home Is Where the Hart Is", "retain_candidate", [],
  "The film credits Leslie Nielsen, whose biography gives the February 11 birth date and says he appeared in more than 100 films.", None,
  [("before_action", "Home Is Where the Hart Is", "It stars Leslie Nielsen"), ("before_action", "Leslie Nielsen", "11 February"), ("before_action", "Leslie Nielsen", "more than 100 films")]),
 (194, "How many people live", "repair_candidate", ["time_unanchored_population"],
  "The CBRL article locates its British Institute in Amman, and the Amman article contains the stated population. The present-tense question lacks an as-of date, so it cannot support a current population claim. The source itself has no date from which to infer one.",
  "Proposed source-scoped wording: According to the supplied Amman article, what population does it report for the city housing CBRL's British Institute? Keep 4,007,526 only after paired-state replay and independent review; do not invent an as-of year.",
  [("before_action", "Council for British Research in the Levant", "the British Institute in Amman (BIA) in Amman, Jordan"), ("before_action", "Amman", "The city has a population of 4,007,526")]),
 (195, "How many miles long", "retain_candidate", [],
  "The Swinton–Doncaster line connects to the East Coast Main Line, whose article explicitly gives its London–Edinburgh length as 393 miles.", None,
  [("before_action", "Swinton–Doncaster line", "on the East Coast Main Line"), ("before_action", "East Coast Main Line", "a 393 mi major railway link between London and Edinburgh")]),
 (196, "George Downer's partner", "retain_candidate", [],
  "George Downer's partner in G & J Downer is identified as Sir John Downer, and John's article states he was South Australia's premier twice.", None,
  [("before_action", "George Downer", "a partner with his brother Sir John Downer in the legal firm G & J Downer"), ("before_action", "John Downer", "was the Premier of South Australia twice")]),
 (197, "From how many sources", "retain_candidate", [],
  "The Alberta article says its constitution is modeled on Britain's; the UK constitution article says it primarily draws from four sources. The question asks about the model, rather than the number of Alberta sources.", None,
  [("before_action", "Constitution of Alberta", "like Britain's (on which it is modeled)"), ("before_action", "Constitution of the United Kingdom", "primarily draws from four sources")]),
 (198, "How many Grand Prix", "retain_candidate", [],
  "Castel d'Ario identifies Tazio Nuvolari as a native, and his biography explicitly includes 24 Grand Prix victories.", None,
  [("before_action", "Castel d'Ario", "birthplace of car race driver Tazio Nuvolari"), ("before_action", "Tazio Nuvolari", "included 24 Grands Prix")]),
 (199, "How much older", "retain_candidate", ["rounded_age_difference"],
  "Blanche Barrow is Buck's wife and Buck is Clyde's brother. Blanche's article explicitly says he was eight years older; the birth dates make this an approximate whole-year description rather than an exact elapsed eight years.", None,
  [("before_action", "Blanche Barrow", "wife of Clyde Barrow's brother Buck"), ("before_action", "Blanche Barrow", "He was 8 years older"), ("before_action", "Buck Barrow", "March 14, 1903")]),
 (200, "How many votes did Halina", "repair_candidate", ["vote_source_causality"],
  "Olendzka received 5,386 votes while standing on the Law and Justice list, and the party article establishes PiS as its abbreviation. 'Votes from the PiS' misleadingly implies the party cast or supplied those votes.",
  "Proposed question: How many votes did Halina Olendzka receive when she ran from the Law and Justice (PiS) list in the 2005 Sejm election? Retain 5,386 only after all same-question states are checked and the revised prompt independently reviewed.",
  [("before_action", "Halina Olendzka", "getting 5386 votes in 33 Kielce district, candidating from the Law and Justice list"), ("before_action", "Law and Justice", "abbreviated to PiS")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-020.jsonl")
