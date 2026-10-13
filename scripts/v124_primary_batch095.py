"""Primary examination of v122 training-supplement candidate states 161–170."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (161, "What are the two roles of the writer", "retain_candidate", ["query_uses_visible_song_writer", "retrieval_adds_roles"],
  "The initial song article names Stephen Malkmus as writer. The query reproduces that named relation and retrieves his biography, which supplies lead singer and guitarist; sentence copying alone does not invalidate a grounded search.", None,
  [("before_action", "Father to a Sister of Thought", "written by Stephen Malkmus of Pavement"), ("retrieved_after_search", "Stephen Malkmus", "lead singer and guitarist")]),
 (162, "What are the two roles of the writer", "retain_candidate", [],
  "The song excerpt identifies writer Stephen Malkmus, and his retrieved biography explicitly calls him the lead singer and guitarist of Pavement. Both requested roles are supported.", None,
  [("before_action", "Father to a Sister of Thought", "written by Stephen Malkmus of Pavement"), ("before_action", "Stephen Malkmus", "lead singer and guitarist")]),
 (163, "David Bernstein was the former chairman", "retain_candidate", ["answer_name_visible_before_search", "search_checks_founder_date"],
  "The executive article already names French Connection as Bernstein's former chairmanship. The query retains that identity and retrieval supplies Stephen Marks's founder relation and early-1970s founding. It verifies the question's qualifier, rather than discovering the answer string.", None,
  [("before_action", "David Bernstein (executive)", "former chairman of French Connection"), ("retrieved_after_search", "French Connection (clothing)", "Founded in the early 1970s by Stephen Marks")]),
 (164, "David Bernstein was the former chairman", "retain_candidate", [],
  "French Connection is linked to former chairman David Bernstein and the early-1970s founding by Stephen Marks in separate action-time documents. The answer is supported, though the name was visible before search.", None,
  [("before_action", "David Bernstein (executive)", "former chairman of French Connection"), ("before_action", "French Connection (clothing)", "Founded in the early 1970s by Stephen Marks")]),
 (165, "Who worked on the cover art for a Rival Sons album", "retain_candidate", ["entity_query_from_visible_candidate", "retrieval_adds_cover_credit"],
  "The initial Thorgerson biography lists both Peter Gabriel and Rival Sons among artists for whom he worked, but does not establish cover-art work on a Rival Sons album. Searching his visible name retrieves the Pressure & Time cover credit.", None,
  [("before_action", "Storm Thorgerson", "Peter Gabriel"), ("before_action", "Storm Thorgerson", "Rival Sons"), ("retrieved_after_search", "Pressure &amp; Time", "cover artwork was designed by Storm Thorgerson")]),
 (166, "Who worked on the cover art for a Rival Sons album", "retain_candidate", [],
  "The album article credits Storm Thorgerson with cover artwork for a Rival Sons record, and his biography includes Peter Gabriel among artists for whom he worked. The two relations are separately supported.", None,
  [("before_action", "Pressure &amp; Time", "cover artwork was designed by Storm Thorgerson"), ("before_action", "Storm Thorgerson", "Peter Gabriel")]),
 (167, "The Unanswered Question is a ballet", "retain_candidate", ["query_uses_visible_ives_identity", "retrieval_adds_epithet"],
  "The ballet and music-work excerpts both identify Charles Ives. The query includes Ives's name and a visible description of his music; retrieval reaches his biography, where the quoted epithet occurs. The query is not especially focused but it uses an established entity and obtains the missing attribute.", None,
  [("before_action", "The Unanswered Question (ballet)", "a ballet made by Eliot Feld to Charles Ives'"), ("retrieved_after_search", "Charles Ives", 'regarded as an "American original"')]),
 (168, "The Unanswered Question is a ballet", "retain_candidate", [],
  "The ballet article ties the work to Charles Ives, and the retrieved biography says he came to be regarded as an American original. The answer's wording matches the quoted epithet.", None,
  [("before_action", "The Unanswered Question (ballet)", "a ballet made by Eliot Feld to Charles Ives'"), ("before_action", "Charles Ives", 'regarded as an "American original"')]),
 (169, "Which show written by Kenn Scott", "retain_candidate", ["query_uses_visible_written_show_list", "retrieval_adds_director"],
  "The query reproduces the tail of the visible list of shows written by Kenn Scott, including Dino Dan. The retrieval finds Dino Dan's page, which directly identifies J. J. Johnson as director. This one success does not establish that a broad list query is robust, but the action-time chain is grounded.", None,
  [("before_action", "Kenn Scott", '"Delilah and Julius", "Dino Dan" and "Doki"'), ("retrieved_after_search", "Dino Dan", "directed by J. J. Johnson")]),
 (170, "Which show written by Kenn Scott", "retain_candidate", [],
  "Kenn Scott's article lists Dino Dan among shows he wrote for, and the show's article identifies J. J. Johnson as director. The requested overlap is supported without confusing Kenn Scott the writer with the same-name actor.", None,
  [("before_action", "Kenn Scott", '"Dino Dan"'), ("before_action", "Dino Dan", "directed by J. J. Johnson")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-095.jsonl", cohort="v122_train_candidates")
