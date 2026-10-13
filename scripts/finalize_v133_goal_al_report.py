"""Primary one-pass AL adjudication: keyed rulings and token-exact citations.

Only reads existing inputs and previously located source bodies. No retrieval,
generation, training, or historical score rewrite.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
import statistics

import run_v133_goal_al_wide_context as al
from transformers import AutoProcessor
from prepare_v133_goal_ah import documents

OUT = al.OUT
RULINGS = {
 "musique:2hop__864101_135373": ("success", "Bering is an unambiguous surname answer for the explicitly named Vitus Bering; both required relations are visible. Neither arm is penalized merely for an equivalent shorter name."),
 "musique:2hop__478978_124498": ("success", "The album is by Matthieu Chedid and his biography names Louis Chedid as his father; no extra song-version requirement is introduced."),
 "musique:2hop__862994_69048": ("unsupported_by_visible_evidence", "Gloria was born in Delta State; the returned chief-judge statement is for Rivers State, not Delta State. Missing target-jurisdiction evidence is not automatically a factual error."),
 "musique:2hop__829358_754719": ("factual_error", "Darley was born in Pedro Leopoldo; the visible Chico Xavier sentence places that city in Minas Gerais. Minsk Region is unrelated and contradicts the supported location chain."),
 "musique:2hop__789565_416401": ("factual_error", "Huang is Wang Yanjun's mother; the Lady Li biography identifies Wang Yanjun as Wang Jipeng's father. Lady Li is Wang Jipeng's cousin, not the required child."),
 "musique:2hop__506035_325512": ("success", "Smith was born in Poplarville, explicitly in Pearl River County. Pearl River is an unambiguous county abbreviation in this question; County need not be repeated."),
 "musique:2hop__337223_56873": ("unsupported_by_visible_evidence", "The team edge is visible, but no returned body supplies Nottingham Forest's FA Cup victory years. 2015-16 describes competition rules or another team, not the required victory relation."),
 "musique:2hop__796221_18803": ("no_answer_repeated_action", "The same query was repeated and no answer was produced; the April 2010 announcement edge is not visible."),
 "musique:2hop__151750_141308": ("success", "The album was released on Apple Records, whose body says it is a division of Apple Corps Ltd. The Ltd suffix is equivalent."),
 "musique:2hop__418360_67370": ("success", "The book names Bill Maher and the show's visible season table states January 19, 2018 as first aired in 2018."),
 "musique:2hop__96194_78303": ("unsupported_by_visible_evidence", "The agent searches Sheree J. Wilson and answers her character Alex Cahill; the necessary film-star identity was not returned. The cast binds Chuck Norris to Cordell Walker, but the movie-to-star link is absent."),
 "musique:2hop__703493_162341": ("incomplete_required_qualifier", "The director is Hoffman, and the visible recognition is a nomination for Best Supporting Actor. Academy Award alone omits nomination status and category; this is incomplete, not a proven false award claim."),
 "musique:2hop__158275_35739": ("success", "The Windows 8 body itself identifies Firefox as a browser and dates the first availability of its version to September 2013; both edges may be supplied by one body."),
 "musique:2hop__435087_55984": ("success", "The song body directly identifies Brandi Carlile as performer and Phil Hanseroth as writer."),
 "musique:2hop__109242_113442": ("success", "The painting is by John Everett Millais, whose visible biography gives the life-date endpoint 13 August 1896."),
 "musique:2hop__69765_43945": ("success", "The pamphlet is by Thomas Paine and the biography explicitly says it crystallized the rebellious demand for independence from Great Britain."),
 "2wiki:982f8e440bdb11eba7f7acde48001122": ("success", "American Scientist was published by Sigma Xi, whose founding year is explicitly 1886."),
 "2wiki:67ecf3100bdc11eba7f7acde48001122": ("success", "The film was directed by Yurek Bogayevicz; the returned director biography explicitly gives Poznan, including the spelling/name variant."),
 "2wiki:3a3d60020bdd11eba7f7acde48001122": ("success", "The song identifies Dominican singer Anais; her biography places her origin in Santo Domingo, Dominican Republic. The question asks where she is from, not proof of birth from a from statement."),
 "2wiki:ef9d40a40bdb11eba7f7acde48001122": ("success", "Armless was directed by Habib Azar, whose biography explicitly calls him American; this is equivalent to the requested United States nationality."),
 "2wiki:5dac1e8e0bdd11eba7f7acde48001122": ("success", "The Smiling Ghost was directed by Lewis Seiler and he was born in New York, New York; the more specific name is supported."),
 "2wiki:1dd33f080bde11eba7f7acde48001122": ("success", "Nicolaus is Christoph's eldest son and the father's biography explicitly says born at Arnstadt. B incorrectly changes to Eisenach, the son's probable birthplace and the father's employment/death location."),
 "2wiki:764039340bde11eba7f7acde48001122": ("unsupported_by_visible_evidence", "A's final input loses both the father binding and the death-place sentence; Edinburgh is not supported as the father's death place. B retains both and answers the supported Hyde Park, Westminster."),
 "2wiki:7bda387a0bdb11eba7f7acde48001122": ("success", "Xavier's body names his father Gerald Tanner, whose biography gives birth on 24 August 1921."),
 "2wiki:f299dd600bdd11eba7f7acde48001122": ("unsupported_by_visible_evidence", "A lacks the 5th-to-4th Duke father sentence and answers 1544, the 5th Duke's birth year. The 4th Duke death endpoint 1575 is already visible in A. B restores the father link and full chain, but repeats its query rather than answering."),
 "2wiki:49d965aa0bde11eba7f7acde48001122": ("success", "The song names Yoyoy Villame, whose biography says he died at Las Pinas Medical Center. A supported hospital is a sufficient answer to where, without imposing city-only granularity."),
 "2wiki:ee3233920bda11eba7f7acde48001122": ("success", "William is explicitly the son of Francesco I Crispo, whose biography says died 1397; Francesco II is not substituted."),
 "musique:2hop__498251_126089": ("unsupported_by_visible_evidence", "Laren is in North Holland; the returned Bonnet body only identifies an artist who died in Laren, not the person in charge. The authority edge is missing."),
 "musique:2hop__530579_13529": ("success", "Sinagra is Maradona's son; the FC Barcelona body dates Maradona's signing to June 1982."),
}

# Minimal body quotations chosen during review, not inferred from case positions.
# Each list corresponds to the two frozen edge IDs in their explicit ledger order.
QUOTES = {
 "musique:2hop__864101_135373": [("b46a2b7fc4839a6a1bbb9d87", "The entire island's natural scenery and wildlife is protected as it is part of the Bering Sea unit of the Alaska Maritime National Wildlife Refuge."), ("11c75d69418f832f2c81e009", "The Bering Sea is named for Vitus Bering, a Danish navigator in Russian service")],
 "musique:2hop__478978_124498": [("7c14b36bf74f5a829a03199e", "Je dis aime (1999) is the second studio album by French singer-songwriter Matthieu Chedid"), ("1ad7b67aa339825a0bb16dc3", "Matthieu Chedid was born in Boulogne-Billancourt, Hauts-de-Seine, France, the son of French singer Louis Chedid")],
 "musique:2hop__829358_754719": [("66b9d6622ad49cead48a46dd", "born 15 December 1989 in Pedro Leopoldo"), ("27701255ad2329b137d47657", "Xavier was born in the city of Pedro Leopoldo, State of Minas Gerais")],
 "musique:2hop__789565_416401": [("63cd6d2dc0aa83a2a91e0321", "the mother of Wang Yanjun (also known as Wang Lin, Emperor Huizong)"), ("837be0cdec2b788fe19d6f9c", "Lady Li was a cousin to Wang Jipeng — her mother was a sister to his father Wang Yanjun")],
 "musique:2hop__506035_325512": [("c82f9ec81610a8767f8bdc0e", "Smith was born in Poplarville, Mississippi"), ("92932c263dc20e1cba507907", "Poplarville is a city in Pearl River County, Mississippi, United States.")],
 "musique:2hop__151750_141308": [("35befcb867cf2e61b47097d1", "Magic Christian Music is the debut album by the British rock band Badfinger, released in early 1970 on Apple Records."), ("74287ea7978b30400a23ebb9", "Apple Records is a record label founded by the Beatles in 1968 as a division of Apple Corps Ltd.")],
 "musique:2hop__418360_67370": [("e66fd2796907e3dd79a70038", "is a book by Bill Maher."), ("540e70832bef0c0591e7bc05", "16 TBA January 19, 2018 (2018 - 01 - 19) TBA")],
 "musique:2hop__703493_162341": [("83c886c3a6cb69ce14f2ad91", "Jack Goes Boating is a 2010 American romantic comedy film directed by Philip Seymour Hoffman"), ("bac49d947f636c1889289acf", "Hoffman was nominated for an Academy Award for Best Supporting Actor.")],
 "musique:2hop__158275_35739": [("a520068d7d13cb2073147041", "The developers of both Chrome and Firefox committed to developing Metro-style versions of their browsers"), ("a520068d7d13cb2073147041", "Firefox's version (which was first made available on the \"Aurora\" release channel in September 2013)")],
 "musique:2hop__435087_55984": [("c6b419eeacd8b728dc17dd38", "is a song released as a single by American folk rock singer Brandi Carlile, written by Phil Hanseroth"), ("c6b419eeacd8b728dc17dd38", "is a song released as a single by American folk rock singer Brandi Carlile, written by Phil Hanseroth")],
 "musique:2hop__109242_113442": [("8577b7b9e1d41bd86fd805c9", "The Blind Girl (1856) is a painting by John Everett Millais"), ("b5e1f1723ba282101a068901", "Sir John Everett Millais, 1st Baronet ( , ; 8 June 1829 – 13 August 1896)")],
 "musique:2hop__69765_43945": [("0cdd78e93c1a71a36c92cfda", "Common Sense is a pamphlet written by Thomas Paine in 1775 -- 76 advocating independence from Great Britain"), ("f2b2c07c44a012550ccb4c5f", "his powerful pamphlet Common Sense (1776), proportionally the all - time best - selling American title, which crystallized the rebellious demand for independence from Great Britain.")],
 "2wiki:982f8e440bdb11eba7f7acde48001122": [("5fd849a3c960626ee3579af9", "magazine published since 1913 by Sigma Xi, The Scientific Research Society."), ("f1c47510d76e5e8992c8df11", "was founded in 1886 at Cornell University")],
 "2wiki:67ecf3100bdc11eba7f7acde48001122": [("9e80bf139ea31929f72c0495", "Edges of the Lord is an award-winning 2001 film, written and directed by Yurek Bogayevicz"), ("ba0902cc761f47e184e6f548", "Yurek Bogajevicz (born Jerzy Bogajewicz in Poznań, 1948)")],
 "2wiki:3a3d60020bdd11eba7f7acde48001122": [("d8e24361464b45394747eb90", "is a song by Dominican singer Anaís."), ("2029cfb2c3f55f134b1bfad0", "is a Latin Grammy Award-nominated singer from Santo Domingo, Dominican Republic.")],
 "2wiki:ef9d40a40bdb11eba7f7acde48001122": [("daae7c3f344e8ef42fc20cee", "Armless is a 2010 comedy film directed by Habib Azar"), ("74ef23844d4985c143d5cb79", "Habib Azar (born October 19, 1979 in Pennsylvania, United States) is an American film, theater and television director")],
 "2wiki:5dac1e8e0bdd11eba7f7acde48001122": [("0e66d9d7f5cb974e517e3f5b", "The Smiling Ghost is a 1941 film directed by Lewis Seiler"), ("d3c2b3bec0a8a4e02b766844", "He was born in New York, New York and died in Hollywood, California.")],
 "2wiki:1dd33f080bde11eba7f7acde48001122": [("4daf04bbbc3ecccdf9a4fd0f", "Johann Nicolaus was the eldest son of Johann Christoph Bach"), ("37f133d1958d1a8bc640b31d", "He was born at Arnstadt, the son of Heinrich Bach")],
 "2wiki:764039340bde11eba7f7acde48001122": [("56f3b855da277556ca88965f", "Abercrombie was the illegitimate son of James Hamilton, 4th Duke of Hamilton."), ("de579d6e8297b9ce0bbbf9a9", "He died on 15 November 1712 as the result of a celebrated duel in Hyde Park, Westminster")],
 "2wiki:7bda387a0bdb11eba7f7acde48001122": [("be72109f7bbbe7707d5b778f", "Tanner's father Gerald Tanner played one game for Richmond in 1941."), ("61f913aa8974b8ffadc6fccd", "Gerald Tanner (born 24 August 1921)")],
 "2wiki:f299dd600bdd11eba7f7acde48001122": [("9041c550da7e2c17d1e61654", "He was the son of Don Juan de la Cerda, 4th Duke of Medinaceli and of Joana Manuel"), ("083960bde826bfe3d1196f74", "Juan de la Cerda, 4th Duke of Medinaceli (c. 1514 – 1575)")],
 "2wiki:49d965aa0bde11eba7f7acde48001122": [("b10a48621b3137439fb4eb9c", "is a novelty song by Yoyoy Villame"), ("4cbda48c1f51a26640d69cbf", "He died of cardiac arrest at the Las Piñas Medical Center in Manila on May 18, 2007.")],
 "2wiki:ee3233920bda11eba7f7acde48001122": [("cfb62cdce1737f2487c65e4f", "He was the son of the tenth Duke Francesco I Crispo"), ("c727639725e20ec156d8b5d6", "Francesco I Crispo, Patrizio Veneto (died 1397)")],
 "musique:2hop__530579_13529": [("f2844a29513a183938ce2a60", "He is the son of Argentine national footballer Diego Maradona."), ("7ecdd37c3432e065ef2ba3cf", "In June 1982, Diego Maradona was signed for a world record fee of £5 million from Boca Juniors. In the following season, under coach Luis, Barcelona won the Copa del Rey, beating Real Madrid. However, Maradona's time with Barcelona was short-lived and he soon left for Napoli.")],
}
MISMATCH = {"musique:2hop__862994_69048", "musique:2hop__829358_754719", "musique:2hop__789565_416401", "musique:2hop__337223_56873", "musique:2hop__96194_78303", "musique:2hop__498251_126089"}


def cite(ids, tokenizer, document_id, quote):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise RuntimeError("actual input tokenizer roundtrip mismatch")
    for doc in documents(text):
        if doc["document_id"] != document_id or quote not in doc["text"]:
            continue
        body = json.dumps(doc["text"], ensure_ascii=False)[1:-1]
        escaped = json.dumps(quote, ensure_ascii=False)[1:-1]
        body_start = text.find(body, *doc["input_char_span"])
        if body_start < 0: raise RuntimeError("body mapping mismatch")
        start = body_start + body.index(escaped)
        end = start + len(escaped)
        indices = [i for i, (s, e) in enumerate(encoded["offset_mapping"]) if s < end and e > start]
        if not indices: raise RuntimeError("empty quote tokens")
        return dict(document_id=document_id, title=doc["title"], quote=quote, body_char_span=[doc["text"].index(quote),doc["text"].index(quote)+len(quote)],
            input_char_span=[start,end], input_token_span=[min(indices),max(indices)+1], tool_block=doc["tool_block"],
            actual_body_sha256=hashlib.sha256(doc["text"].encode()).hexdigest())
    return None


def main():
    al.guard("awaiting_one_pass_semantic_review")
    processor = AutoProcessor.from_pretrained(str(al.assets.BASE), local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    packets = al.rows(OUT / "review-packets.jsonl")
    if len(packets) != 29 or set(RULINGS) != {p["case_id"] for p in packets}:
        raise RuntimeError("29 keyed rulings required")
    data = {arm: al.rows(OUT / f"trajectory-{arm}.jsonl") for arm in ("A", "B")}
    manifest = al.read(OUT / "manifest.json")
    reviews = []
    seen = {}
    for p in packets:
        case_id = p["case_id"]
        by_arm = {}
        for arm in ("A", "B"):
            state = p[arm]
            category, reason = RULINGS[case_id]
            if case_id == "2wiki:1dd33f080bde11eba7f7acde48001122" and arm == "B": category="factual_error"
            if case_id == "2wiki:764039340bde11eba7f7acde48001122" and arm == "B": category="success"
            if case_id == "2wiki:f299dd600bdd11eba7f7acde48001122" and arm == "B": category="no_answer_repeated_action"
            citations = []
            for i, edge in enumerate(p["edges"]):
                doc_id, quote = QUOTES.get(case_id, [(e["document_id"], e["quote"]) for e in p["edges"]])[i]
                found = cite(state["token_ids"], tokenizer, doc_id, quote)
                citations.append(dict(edge_id=edge["edge_id"], subject=edge["subject"], relation=edge["relation"], object=edge["object"],
                    actual_visible=found is not None, citation=found))
            complete = all(c["actual_visible"] for c in citations)
            if category=="success" and not complete:
                raise RuntimeError(f"success without verified actual-input chain: {case_id}:{arm}")
            key = hashlib.sha256(json.dumps([p["question"], state["token_ids"], state["answer"]], ensure_ascii=False).encode()).hexdigest()
            prior = seen.get(key)
            if prior is not None and prior["category"] != category: raise RuntimeError("same combination got different rulings")
            record = dict(case_id=case_id, arm=arm, question=p["question"], answer=state["answer"], category=category, reason=reason,
                complete_chain_visible=complete, edges=citations, combination_sha256=key,
                reused_same_combination=prior is not None, training_use_forbidden=True,
                entity_relation_mismatch=case_id in MISMATCH or case_id=="2wiki:1dd33f080bde11eba7f7acde48001122" and arm=="B" or case_id=="2wiki:f299dd600bdd11eba7f7acde48001122" and arm=="A")
            seen[key]=record
            by_arm[arm]=record
        reviews.append(dict(case_id=case_id, A=by_arm["A"], B=by_arm["B"]))
    for r in reviews: al.append(OUT / "semantic-review.jsonl", r)
    metrics = al.read(OUT / "metrics.json")["metrics"]
    success = {arm: {r["case_id"] for r in reviews if r[arm]["category"]=="success"} for arm in ("A","B")}
    counts = {arm: Counter(r[arm]["category"] for r in reviews) for arm in ("A","B")}
    errors = {arm: counts[arm]["factual_error"]+counts[arm]["unsupported_by_visible_evidence"] for arm in ("A","B")}
    maps = {arm: {r["case_id"]:r for r in data[arm]} for arm in ("A","B")}
    protected_lost = [c for c in manifest["protected"] if maps["A"][c]["em"] and not maps["B"][c]["em"]]
    for arm, items in data.items():
        calls=[t for r in items for t in r["trace"]]
        metrics[arm]["overbudget_search_attempts"]=sum(t.get("action")=="search" and t["step"]>=manifest["arms"][arm]["max_searches_per_case"] for t in calls)
        metrics[arm]["ordinary_invalid_or_repeated"]=sum(r["reason"] in ("invalid_action","repeated_action") for r in items)
    gates=dict(supported_net_plus_two=len(success["B"])-len(success["A"])>=2,
        factual_plus_unsupported_not_higher=errors["B"]<=errors["A"], invalid_or_repeated_not_higher=metrics["B"]["invalid_or_repeated"]<=metrics["A"]["invalid_or_repeated"],
        protection_losses_at_most_one=len(protected_lost)<=1)
    decision=dict(goal="AL", status="complete", config=manifest["arms"], counts=al.read(OUT / "counts.json"), metrics=metrics,
        supported_success={arm:len(success[arm]) for arm in success}, categories={arm:dict(c) for arm,c in counts.items()},
        errors=errors, supported_gained=sorted(success["B"]-success["A"]), supported_lost=sorted(success["A"]-success["B"]),
        complete_chain_visible={arm:sum(r[arm]["complete_chain_visible"] for r in reviews) for arm in ("A","B")},
        entity_relation_mismatch={arm:sum(r[arm]["entity_relation_mismatch"] for r in reviews) for arm in ("A","B")},
        protected_lost=protected_lost, gates=gates, passed=all(gates.values()), review_unique_combinations=len(seen),
        review_maximum=58, fixed_denominator=64, semantic_denominator=29, unconfirmed=35,
        semantic_scoring="AL primary one-pass keyed review; equivalent surnames/counties accepted symmetrically; old scores preserved",
        next_route="Stop this wide-budget upgrade route; prepare one bounded entity-role and answer/search-decision intervention under complete visible evidence, no new execution in AL.",
        model_sha256=al.assets.EXPECTED_ADAPTER, released=False)
    if len(al.rows(OUT / "generation-journal.jsonl")) != decision["counts"]["generations"]:
        raise RuntimeError("physical journal count mismatch")
    if decision["counts"]["generations"]!=10+sum(len(r["trace"]) for items in data.values() for r in items):
        raise RuntimeError("generation population mismatch")
    if decision["counts"]["searches"]!=sum(r["searches"] for items in data.values() for r in items):
        raise RuntimeError("physical search mismatch")
    al.write(OUT / "decision.json", decision)
    changes=[]
    for r in reviews:
        if r["A"]["category"] != r["B"]["category"] or r["A"]["complete_chain_visible"] != r["B"]["complete_chain_visible"]:
            changes.append(r)
    al.write(OUT / "mechanism-changes.json", changes)
    shared=[c for c in maps["A"] if maps["A"][c].get("elapsed_seconds") is not None and maps["B"][c].get("elapsed_seconds") is not None]
    latency={arm:statistics.mean(maps[arm][c]["elapsed_seconds"] for c in shared) for arm in ("A","B")}
    al.write(OUT / "matched-latency.json", dict(cases=len(shared),mean_question_seconds=latency))
    text=f"""# Goal AL：扩大窗口消除了裁剪，但没有净搜索收益

## 结果

固定Goal Z完整轨迹seed42和原提示：A为2048输入/128输出，B为16384输入/2048输出。EM从24/64降为23/64，F1从{metrics['A']['F1']:.4f}降为{metrics['B']['F1']:.4f}。29题一次同口径审核中，两臂证据支持成功均19/29；新增1题、丢失1题，净增0。继续门槛未通过，不发布大窗口配置，不启动蒸馏或回归。

| 指标 | A | B |
|---|---:|---:|
| EM /64 | 24 | 23 |
| 证据支持成功 /29 | 19 | 19 |
| 完整链在最终动作输入可见 /29 | {decision['complete_chain_visible']['A']} | {decision['complete_chain_visible']['B']} |
| 事实错误+证据不支持 /29 | {errors['A']} | {errors['B']} |
| 无效、重复或预算外动作题 /64 | {metrics['A']['invalid_or_repeated']} | {metrics['B']['invalid_or_repeated']} |
| 普通解析无效或重复题 /64 | {metrics['A']['ordinary_invalid_or_repeated']} | {metrics['B']['ordinary_invalid_or_repeated']} |
| 第4次搜索尝试 /64 | {metrics['A']['overbudget_search_attempts']} | {metrics['B']['overbudget_search_attempts']} |
| 正文裁剪调用 | 4 | 0 |
| 搜索数 | 130 | 130 |
| 模型调用 | 194 | 194 |
| 最大实际输入 | 1997 | 2631 |
| 最大实际输出 | 29 | 29 |
| 输出长度停止 | 0 | 0 |
| 共同63题平均秒 | {latency['A']:.3f} | {latency['B']:.3f} |

## 赢回、丢失与机制

- 赢回`2wiki:764039340bde11eba7f7acde48001122`：两臂查询和原始返回相同。A的2245-token原输入裁成840，父亲绑定和死亡地点句都丢失，回答Edinburgh；B完整保留，回答Hyde Park, Westminster。严格EM不接受额外地域词，因此此收益出现在支持评分中。逐边引文及token位置见semantic-review.jsonl。
- 丢失`2wiki:1dd33f080bde11eba7f7acde48001122`：A在791-token输入中正确回答父亲出生地Arnstadt；B在2104-token完整输入中仍含相同必要链，但把儿子可能出生的Eisenach、或父亲任职/死亡地Eisenach套到了父亲出生地。它证明保留更多正文可能增加实体/关系干扰，不能证明模型权重提高。
- `2wiki:f299dd600bdd11eba7f7acde48001122`：A父亲绑定被删除，但第4代公爵1575死亡年份的首段已经可见；B恢复父亲绑定及完整链，却重复查询并停止无答案。此前父亲边账本误引了婚姻句，本轮使用实际正文中的He was the son...，旧账本不改。
- 未准入`2wiki:91c13a780bd911eba7f7acde48001122`：输入由1262恢复至2631，回答仍为Seleucid Empire；保持来源未确认，不纳入支持成功。
- Chelsea训练题工程哨兵：245恢复至2363 token，完整保留2012 Roberto Di Matteo冠军联赛行，模型仍回答Rafa Benítez（表中对应2013 Europa League），未恢复正确回答。它不进入64题分母或产品成功。

工程回放仅作条件诊断，真实搜索收益只来自两臂64题。7题连续搜索保护题未丢失。实际输出均少于128 token，没有旧输出上限截断；记录的首次分歧均发生在已有超预算前缀处。联合预算实验的观察支持输入保留改变了这些动作，不能外推为16384长度推理能力：本轮最长实际输入只有2631。

## 评分边界与配置修复

本轮不继承AK的17/29标签。Bering在命名为Vitus Bering的唯一人物上下文中接受姓氏简称；Pearl River接受题目所问县的简称；Academy Award缺提名及Best Supporting Actor仍记限定不足。Gloria和Hero病例缺目标关系，不因金标差异自动判事实错。旧分数和审核保持原样。本轮主agent审核仍非独立人工金标准。

模型配置262144位置；推理长度集中在单份config.json，Goal Q工厂、rollout和报告读取相同设置并校验漂移。CLI、旧实验和训练默认值不进入本轮路径。保持legacy算法、默认产品及历史脚本不变。

## 执行与限制

本轮物理生成398/522（工程10、A194、B194），真实搜索260/384，无训练、新确认或发布。首次输入重建错误和可选generation_config文件读取错误均发生于零生成准备阶段，现场另存；依据用户“修复完继续”指令完成受控恢复。A第一题4次生成后的答案字段映射错误仅在后处理中发生，已通过保存token/action核销，重复生成0；其1题/4调用延迟缺失，不用历史值填补，比较共同63题。

下一步不再扩窗口凑收益，也不继续全面审核。只准备完整可见证据下的实体/角色绑定与停止搜索决策实验；优先针对Johann、Darley、Huang及公爵/Chelsea等已保存反例，评价题仅作诊断，不转入训练。任何训练仍需独立训练材料及匹配对照，本轮不启动。
"""
    (OUT / "REPORT.md").write_text(text,encoding="utf-8")
    al.write(OUT / "status.json", dict(status="complete", passed=decision["passed"], generations=decision["counts"]["generations"],searches=decision["counts"]["searches"]))
    al.write(OUT / "delivery-checksums.json", {p.name:al.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="delivery-checksums.json"})
    print(json.dumps(decision,ensure_ascii=False))


if __name__=="__main__": main()
