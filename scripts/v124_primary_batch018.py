"""Primary case examination of original training rows 171–180."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (171, "The actress that played Jane Bennet", "retain_candidate", ["search_verifies_cross_film_actor"],
  "The initial Jack Reacher film article gives McQuarrie, Tom Cruise, Rosamund Pike and the source novel One Shot, but not Pike's Jane Bennet role. The copied film-cast sentence retrieves Pike's biography confirming that role. The query verifies the missing cross-film identity.", None,
  [("before_action", "Jack Reacher (film)", "based on Lee Child's 2005 novel \"One Shot\""), ("before_action", "Jack Reacher (film)", "with Rosamund Pike"), ("retrieved_after_search", "Rosamund Pike", "portrayed Jane Bennet in \"Pride & Prejudice\" (2005)")]),
 (172, "Which team, controlled", "retain_candidate", [],
  "The stadium article names Spain in the 2015 under-19 final. The Spain under-19 team article explicitly names the Royal Spanish Football Federation as controller. The answer identifies that team.", None,
  [("before_action", "Katerini Stadium", "the final of the 2015 UEFA European Under-19 Championship between Spain and Russia"), ("before_action", "Spain national under-19 football team", "controlled by Royal Spanish Football Federation")]),
 (173, "The Kentucky Democratic", "retain_candidate", [],
  "The 2008 Kentucky Democratic primary article explicitly gives May 20, 2008 and the six-district 34-delegate allocation, matching the question's disambiguating clue. The date is supported.", None,
  [("before_action", "Kentucky Democratic primary, 2008", "took place May 20, 2008"), ("before_action", "Kentucky Democratic primary, 2008", "six congressional districts was awarded all of that district's delegates, totaling 34")]),
 (174, "What is the name of the painting", "repair_candidate", ["unsupported_painting_medium"],
  "Hayden's biography identifies Sky's the Limit as his best-known commission at O'Hare; the airport article places O'Hare on Chicago's Far Northwest Side. The source describes Hayden's work generally as neon art and never calls this commission a painting. The question's medium claim is unsupported.",
  "Proposed question: What is the name of Michael Hayden's best-known commission at the airport on Chicago's Far Northwest Side? Retain the title only after paired-state and independent review.",
  [("before_action", "Michael Hayden (artist)", "best-known commission is \"Sky's the Limit\" (1987) at O'Hare International Airport"), ("before_action", "O'Hare International Airport", "located on the Far Northwest Side of Chicago")]),
 (175, "What American archbishop", "retain_candidate", ["based_on_is_attributed_not_certain"],
  "Gilday's article states that the fictional antagonist is said to be based on Paul Marcinkus. Searching the visible character name retrieves Marcinkus's biography, which calls him an American archbishop. The question's 'said to have been based on' qualification must be preserved.", None,
  [("before_action", "Archbishop Gilday", "said to have been based on Paul Marcinkus"), ("retrieved_after_search", "Paul Marcinkus", "was an American archbishop of the Roman Catholic Church")]),
 (176, "What number president", "repair_candidate", ["unanchored_presidential_ordinal"],
  "The book article identifies Donald Trump as Kelly's feud counterpart and the retrieved biography calls him 45th president in its historical snapshot. The question lacks an office-term or date qualifier; ordinal descriptions of a person can be ambiguous across separate terms. Keep the original 45th label only under an explicit term anchor.",
  "Proposed question: According to the cited presidency beginning January 20, 2017, what numbered U.S. president was the person who feuded with Megyn Kelly in Settle for More? Rebuild this and search row 51 together.",
  [("before_action", "Settle for More", "her feud with then Presidential nominee Donald Trump"), ("before_action", "Donald Trump", "the 45th and current President of the United States, in office since January 20, 2017")]),
 (177, "Vale of Tears", "retain_candidate", [],
  "The novel article identifies Peter T. King as author and politician; King's biography gives April 5, 1944. The year answer is supported and matches the earlier search row 158.", None,
  [("before_action", "Vale of Tears (novel)", "the third novel by Peter T. King"), ("before_action", "Peter T. King", "born April 5, 1944")]),
 (178, "Free Zone followers", "retain_candidate", [],
  "The Free Zone article ties the relevant original teachings to Scientology founder L. Ron Hubbard. Hubbard's biography explicitly says he developed a system called Dianetics. The answer is supported by the two documents.", None,
  [("before_action", "Free Zone (Scientology)", "original teachings of Scientology's founder, L. Ron Hubbard"), ("before_action", "L. Ron Hubbard", "developed a system called Dianetics")]),
 (179, "The first suburban store", "quarantine_pending_repair", ["unanchored_current_occupant", "awkward_question_wording"],
  "The Hollywood Theatre article describes Rite Aid as 'currently' occupying the first suburban Fred Meyer site, but gives no date. The Fred Meyer article verifies the 1922 Portland founding. This is the answer counterpart of quarantined search row 67, and a source-snapshot occupant is not a timeless answer.",
  "Recover dated source provenance for the occupant, then jointly revise both same-question states. Do not infer a present-day occupant from this record.",
  [("before_action", "Hollywood Theatre (Portland, Oregon)", "first suburban Fred Meyer store, which is currently occupied by Rite Aid"), ("before_action", "Fred Meyer", "founded in 1922 in Portland, Oregon")]),
 (180, "Apart from Andy Warhol", "retain_candidate", [],
  "Edwin Daniel's article names Jean-Michel Basquiat as the other major influence. Basquiat's biography calls him an American artist, supporting the nationality answer.", None,
  [("before_action", "Edwin Daniel", "major influences were Andy Warhol and Jean-Michel Basquiat"), ("before_action", "Jean-Michel Basquiat", "was an American artist")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-018.jsonl")
