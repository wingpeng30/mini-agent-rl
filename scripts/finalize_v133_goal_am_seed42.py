"""One-pass primary-agent rulings and delivery, CPU only; no sampling."""
import json
from pathlib import Path
import analyze_v133_goal_am as review

OUT=review.OUT
# Human adjudications keyed by identity, not case order or gold-string hit.
# Each relation has been read in the archived actual tool-body input.
RULINGS={
"musique:2hop__864101_135373":("success","Island belongs to the Bering Sea unit; the sea body explicitly says named for Vitus Bering.",False),
"musique:2hop__478978_124498":("success","Album artist Matthieu Chedid is explicitly the son of Louis Chedid.",False),
"musique:2hop__862994_69048":("unsupported_by_visible_evidence","Birthplace is Delta State; returned Adama Lamikanra relation concerns Rivers State, not the requested birthplace.",True),
"musique:2hop__829358_754719":("factual_error","Darley was born in Pedro Leopoldo; visible city/state relation gives Minas Gerais, not Minsk Region.",True),
"musique:2hop__789565_416401":("factual_error","Huang's son is Wang Yanjun; Lady Li body explicitly binds Wang Jipeng to father Wang Yanjun. Lady Li is his cousin/wife, not the requested child.",True),
"musique:2hop__506035_325512":("success","Smith birthplace Poplarville is explicitly in Pearl River County; County is a valid disambiguating suffix.",False),
"musique:2hop__337223_56873":("unsupported_by_visible_evidence","Forest membership is visible but the necessary Forest FA Cup win dates are absent; 2015-16 cannot be justified by a different club's competition.",True),
"musique:2hop__796221_18803":("success","MV Rapana is a Royal Dutch/Shell tanker; the Shell body explicitly describes seeking a buyer for all Finland operations in April 2010.",False),
"musique:2hop__151750_141308":("success","Magic Christian Music was released on Apple Records, explicitly a division of Apple Corps Ltd; this is label-to-company, not artist signing.",False),
"musique:2hop__418360_67370":("success","Book author is Bill Maher; his Real Time season table gives first airing January 19, 2018.",False),
"musique:2hop__96194_78303":("factual_error","Film star is Chuck Norris and Walker cast binds him to Cordell Walker. Simon Moon/The Terror are the other film's killer, not Norris's TV character.",True),
"musique:2hop__703493_162341":("incomplete_required_qualifier","Hoffman is the director; Charlie Wilson's War body supports a nomination for Best Supporting Actor. B names the category but omits nomination status; A names an event and does not establish receipt of recognition.",True),
"musique:2hop__158275_35739":("success","Firefox Windows 8 Aurora version was first made available in September 2013, explicitly stated in body.",False),
"musique:2hop__435087_55984":("success","Album performer Brandi Carlile and song written by Phil Hanseroth are both explicit.",False),
"musique:2hop__109242_113442":("success","Painting creator John Everett Millais and his death date 13 August 1896 are explicit.",False),
"musique:2hop__69765_43945":("success","Common Sense was written by Paine; his body says it crystallized rebellious demand for independence.",False),
"2wiki:982f8e440bdb11eba7f7acde48001122":("success","Magazine publisher Sigma Xi was founded in 1886; both explicit bodies present.",False),
"2wiki:67ecf3100bdc11eba7f7acde48001122":("success","Film director Yurek Bogayevicz/Bogajevicz is the same referent; biography explicitly born in Poznan.",False),
"2wiki:3a3d60020bdd11eba7f7acde48001122":("success","Song by Dominican singer Anais; biography also identifies Santo Domingo, Dominican Republic. Asked country she is from, not proof of legal citizenship.",False),
"2wiki:ef9d40a40bdb11eba7f7acde48001122":("success","Armless director Habib Azar is explicitly an American director; nationality equivalence allowed.",False),
"2wiki:5dac1e8e0bdd11eba7f7acde48001122":("success","Director Lewis Seiler biography explicitly born in New York, New York.",False),
"2wiki:1dd33f080bde11eba7f7acde48001122":("factual_error","Father Johann Christoph Bach was born at Arnstadt; Eisenach is the son's birthplace.",True),
"2wiki:764039340bde11eba7f7acde48001122":("success","Abercrombie's father James Hamilton died in the duel in Hyde Park, Westminster; adding Westminster preserves the referent.",False),
"2wiki:7bda387a0bdb11eba7f7acde48001122":("success","Xavier's father Gerald Tanner was born 24 August 1921.",False),
"2wiki:f299dd600bdd11eba7f7acde48001122":("success","5th Duke was son of the 4th Duke; 4th Duke body explicitly says death in 1575. Spouse sentence is not father evidence; citation corrected separately.",False),
"2wiki:49d965aa0bde11eba7f7acde48001122":("success","Butse Kik performer Yoyoy Villame died at Las Pinas Medical Center in Manila; candidate uses literal full location.",False),
"2wiki:ee3233920bda11eba7f7acde48001122":("success","William II was son of Francesco I Crispo, whose biography explicitly says died 1397.",False),
"musique:2hop__498251_126089":("unsupported_by_visible_evidence","Laren is in North Holland; no governing-person edge is present. Gooi is an area rather than the requested person.",True),
"musique:2hop__530579_13529":("success","Diego Sinagra is son of Diego Maradona; Barcelona body explicitly records signing in June 1982.",False),
}

def main():
    packets=review.rows(OUT/"review-packets-seed42.jsonl")
    fixes=[
      ("musique:2hop__864101_135373",1,"The Bering Sea is named for Vitus Bering, a Danish navigator in Russian service, who in 1728 was the first European to systematically explore it, sailing from the Pacific Ocean northward to the Arctic Ocean.","The inherited pointer described sea separation, not the naming relation."),
      ("musique:2hop__478978_124498",1,"Matthieu Chedid was born in Boulogne-Billancourt, Hauts-de-Seine, France, the son of French singer Louis Chedid, and the grandson of the Egyptian-born French writer and poet of Lebanese descent Andree Chedid who has written lyrics for him.","The inherited sister sentence did not establish fatherhood."),
      ("musique:2hop__789565_416401",1,"Lady Li was a cousin to Wang Jipeng — her mother was a sister to his father Wang Yanjun (known as Wang Lin during his reign as emperor), and her father was the official Li Min (李敏), who served as a chancellor during Wang Lin's reign.","The inherited marriage/prince sentence did not name the father; explicit father clause exists in the same actual body."),
      ("2wiki:f299dd600bdd11eba7f7acde48001122",0,"He was the son of Don Juan de la Cerda, 4th Duke of Medinaceli and of Joana Manuel, daughter of Sancho de Noronha, 2nd Count of Faro.","The inherited spouse sentence did not establish the requested father."),
    ]
    corrections=[]
    for case,i,quote,reason in fixes:
        p=next(r for r in packets if r["case_id"]==case and r["model"]=="42-A")
        e=p["edges"][i]
        # Preserve original spelling exactly; no normalization of evidence.
        if case=="musique:2hop__478978_124498":
            quote=quote.replace("Andree Chedid","Andrée Chedid")
        corrections.append(dict(case_id=case,edge_id=e["edge_id"],document_id=e["citation"]["document_id"],quote=quote,reason=reason))
    review.write(OUT/"citation-corrections.json",corrections)
    review.correct_citations(42)
    answers={};probes={}
    for arm in ("A","B"):
        label=f"42-{arm}";answers[label]={};probes[label]={}
        for r in packets:
            if r["model"]!=label:continue
            category,reason,mismatch=RULINGS[r["case_id"]]
            answers[label][r["case_id"]]=dict(category=category,reason=reason,entity_relation_mismatch=mismatch)
        for p in review.rows(OUT/"probe-review-packets-seed42.jsonl"):
            if p["model"]!=label or p["category"]!="complete":continue
            case=p["probe_id"].removeprefix("complete:")
            category,reason,_=RULINGS[case]
            probes[label][p["probe_id"]]=dict(correct=category=="success",reason=reason)
    review.write(OUT/"primary-adjudications-seed42.json",dict(answers=answers,probes=probes,
        reviewer="primary agent; exposed sources; one pass; actual input and body only",
        scope="58 answer combinations plus 48 complete-evidence probe answers; 16 missing probes mechanically classified"))
    review.finalize(42)

if __name__=="__main__":main()
