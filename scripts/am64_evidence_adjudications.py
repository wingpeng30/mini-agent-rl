"""Primary-agent keyed offline adjudications, 2026-10-07.

These are explicit semantic judgments, not substring-derived labels. A quote is
an exact body substring. None means the reviewed visible bodies do not establish
the edge; a nearby relation must not be promoted to proof of the requested edge.
Answer correctness is judged against frozen reference semantics, allowing
documented, question-compatible alternatives. It is not external gold validation.
"""

# (subject, relation, object, document title, exact quote, missing-edge rationale)
def edge(subject, relation, obj, title=None, quote=None, missing=None):
    return dict(subject=subject, relation=relation, object=obj, title=title,
                quote=quote, missing_reason=missing)


def judgment(correct, reason, edges, flags=()):
    return dict(answer_correct=correct, answer_reason=reason, edges=edges,
                annotation_flags=list(flags), status="reviewed")


NEW = {
 "musique:2hop__46376_158105": judgment(False, "回答的是表演类型，未回答题目所指的两个操作系统特点。", [
  edge("题目所指人物", "最高收入者（冻结链指定 Bill Gates）", "Bill Gates", missing="可见正文没有 Bill Gates；离线支持正文仅称 richest，不直接证明 highest paid。"),
  edge("Bill Gates", "演示时强调的 Windows 98 特点", "ease of use and enhanced support for Plug and Play", missing="可见输入无 Windows 98 演示及两项特点。")], ["highest_paid_vs_richest", "frozen_reference_source_gap"]),
 "musique:2hop__235471_565925": judgment(True, "Moby Grape 的唱片由 Columbia Records 发行，答案与冻结参考一致；并非把这张合辑的 Sundazed 发行厂牌当作乐队唯一厂牌。", [
  edge("The Place and the Time", "歌曲表演者", "Moby Grape", "The Place and the Time", "songs by Moby Grape"),
  edge("Moby Grape", "唱片所属厂牌", "Columbia Records", "Great Grape", "Great Grape is a compilation album released by Columbia Records in 1972 that compiles songs from three of Moby Grape's Columbia albums")], ["artist_multiple_labels"]),
 "musique:2hop__95673_121319": judgment(False, "Lubumbashi 不是冻结链中 Edward G. Robinson 的就读机构；可见检索偏向 Sammy Baloji。", [
  edge("Sammy Going South", "出演者", "Edward G. Robinson", "Sammy Going South", "starring Edward G. Robinson, Fergus McClelland and Constance Cummings"),
  edge("Edward G. Robinson", "就读机构", "American Academy of Dramatic Arts", missing="Robinson 的教育正文未进入最终输入；所有可见文档均不提供该就读关系。")], ["multiple_film_cast_members"]),
 "musique:2hop__121145_561444": judgment(True, "Rabbi 为尊称，Dovber Schneuri 与冻结答案同一人物。正文把两人分别列为同一 Chabad 运动第二、第三任 Rebbe，可直接推得继任顺序。", [
  edge("Derech Mitzvosecha", "作者", "Menachem Mendel Schneersohn", "Derech Mitzvosecha", "authored by Rabbi Menachem Mendel Schneersohn (1789–1866), the third Rebbe of the Chabad Hasidic movement"),
  edge("Menachem Mendel Schneersohn", "同一运动中继任前一任 Rebbe", "Dovber Schneuri", "Chaya Mushka Schneersohn", "Chaya Mushka Schneersohn was the daughter of Rabbi Dovber Schneuri, the second Rebbe of the Chabad Hasidic movement, and the wife of Rabbi Menachem Mendel Schneersohn the third Rebbe.")], ["ordinal_succession_inference", "honorific_equivalence"]),
 "musique:2hop__44522_58556": judgment(False, "答案只给 FDA 机构名称，没有回答其食品安全法律／系统名称 FSMA。", [
  edge("OTC／处方药上市分类", "决定机构", "FDA", missing="可见正文只授予广义食品药品监管权，没有明确 OTC／处方分类决定权；不能用泛监管权替代此关系。"),
  edge("FDA", "食品安全法律／系统", "Food Safety Modernization Act (FSMA)", "FDA Food Safety Modernization Act", "The FSMA has given the Food and Drug Administration (FDA) new authorities to regulate the way foods are grown, harvested and processed.")], ["question_system_vs_act_wording", "first_edge_source_gap"]),
 "musique:2hop__475070_532404": judgment(False, "English 是语言／国籍形容词，未给出 Charles Tupper 的出生地。", [
  edge("Frances Tupper", "配偶", "Charles Tupper", "Frances Tupper", "was the wife of Sir Charles Tupper, the sixth Prime Minister of Canada"),
  edge("Charles Tupper", "出生地", "Amherst", missing="可见输入无 Charles Tupper 出生地；离线 Arthur Rupert Dickey 的 Amherst 出生句属于另一个人。")], ["mislabeled_support_entity", "frozen_reference_source_gap"]),
 "musique:2hop__199216_33564": judgment(False, "1970s 只覆盖冻结参考年代范围的末段，未完整回答 1950s–1970s；本题参考所述是在 Hyderabad 设立机构，未必等同机构整体成立。", [
  edge("Agni-V", "研发机构（冻结链的 manufacturer）", "Defence Research and Development Organisation", "Agni-V", "Agni-V is an intercontinental ballistic missile developed by the Defence Research and Development Organisation (DRDO) of India."),
  edge("DRDO", "在 Hyderabad 设立的年代范围", "From the 1950s to the 1970s", missing="最终输入无 Hyderabad 设立机构年代正文。")], ["development_vs_manufacture", "local_establishment_vs_founding", "incomplete_time_range"]),
 "musique:2hop__374495_68633": judgment(True, "两侧正文给出 IOC 及 Thomas Bach；只按冻结语料当时的任职快照判定，不作为现在任职信息。", [
  edge("Avery Brundage", "曾任主席的组织", "International Olympic Committee", "Avery Brundage", "was the fifth President of the International Olympic Committee (IOC), serving from 1952 to 1972"),
  edge("International Olympic Committee", "冻结语料快照中的主席", "Thomas Bach", "President of the International Olympic Committee", "Incumbent Thomas Bach since 10 September 2013 International Olympic Committee")], ["historical_corpus_snapshot"]),
 "musique:2hop__463353_66723": judgment(False, "Winona Cargile Alexander 与冻结目标 James Howard Meredith 不同；最终输入缺目标学校首位黑人学生关系。", [
  edge("Swayze Field", "所属学校", "University of Mississippi", "Swayze Field", "Oxford-University Stadium at Swayze Field is the home of the University of Mississippi Rebels college baseball team"),
  edge("University of Mississippi", "首位获录取黑人学生", "James Howard Meredith", missing="可见正文没有 Meredith 的入学关系。")]),
 "musique:2hop__71034_343058": judgment(False, "Jennifer Saunders 是桥接歌手，未回答其配偶 Adrian Edmondson。", [
  edge("Holding Out for a Hero in Shrek 2", "演唱者", "Jennifer Saunders", "Holding Out for a Hero", "Jennifer Saunders recorded the song for the 2004 film Shrek 2."),
  edge("Jennifer Saunders", "配偶", "Adrian Edmondson", missing="Mirrorball 正文只写导演 Adrian Edmondson 与编剧 Jennifer Saunders，不能由共同工作推断婚姻。")], ["labeled_body_lacks_spouse_relation"]),
 "musique:2hop__817096_15463": judgment(False, "P&O European Ferries 来自无关的合并案例，未回答 Compaq 在2002年的合并对象 Hewlett Packard。", [
  edge("CEMM", "开发方", "Compaq", "CEMM", "In 1986, Compaq was the first vendor to ship a PC compatible computer with a 386 CPU, the Deskpro 386, and it was natural for them to develop solutions leveraging the specific features of their new hardware and in this case allowing existing EMS-compatible DOS programs to access all the memory."),
  edge("Compaq", "2002年合并对象", "Hewlett Packard", missing="Dell 正文中的 Compaq–Hewlett Packard 合并句未进入最终输入。")], ["developer_inferred_from_document_context"]),
 "2wiki:91c13a780bd911eba7f7acde48001122": judgment(True, "父亲为 Antiochus I Soter，其统治国家为 Seleucid Empire，与题目及参考一致。", [
  edge("Stratonice of Macedon", "父亲", "Antiochus I Soter", "Stratonice of Macedon", "was the daughter of Stratonice of Syria and of the Seleucid king Antiochus I Soter"),
  edge("Antiochus I Soter", "所属／统治国家", "Seleucid Empire", "Antiochus I Soter", "was a king of the Hellenic Seleucid Empire")]),
 "2wiki:b587d3d20bdd11eba7f7acde48001122": judgment(False, "Sijena 是 Maria 自己的死亡地；其丈夫 Peter 的生卒括注给出死亡地 Pinos Puente。", [
  edge("Maria of Aragon, Lady of Cameros", "丈夫", "Peter of Castile, Lord of Cameros", "Maria of Aragon, Lady of Cameros", "She married Peter of Castile, Lord of Cameros (1290–1319), son of Sancho IV of Castile."),
  edge("Peter of Castile, Lord of Cameros", "死亡地", "Pinos Puente", "Peter of Castile, Lord of Cameros", "Peter of Castile, in Spanish Pedro de Castilla (Valladolid, 1290 – Pinos Puente, 25 June 1319)")]),
 "2wiki:d52aa2a20bdb11eba7f7acde48001122": judgment(True, "母亲姓名及其生卒括注中的出生地 Strasbourg 均可见。", [
  edge("Louise Sophie of Hanau-Lichtenberg", "母亲", "Countess Palatine Anna Magdalena of Birkenfeld-Bischweiler", "Louise Sophie of Hanau-Lichtenberg", "was a daughter of Johann Reinhard II, Count of Hanau-Lichtenberg (1628–1666) and Countess Palatine Anna Magdalena of Birkenfeld-Bischweiler (1640–1693)"),
  edge("Countess Palatine Anna Magdalena of Birkenfeld-Bischweiler", "出生地", "Strasbourg", "Countess Palatine Anna Magdalena of Birkenfeld-Bischweiler", "Countess Palatine Anna Magdalena of Birkenfeld- Bischweiler( 14 February 1640, Strasbourg – 12 December 1693, Babenhausen)")]),
 "2wiki:66616e2a0bde11eba7f7acde48001122": judgment(False, "Żagań-Głogów 在正文中是公爵领地，不能替代父亲的死亡地 Szprotawa。", [
  edge("Wenceslaus of Krosno", "父亲", "Henry VIII the Sparrow", "Wenceslaus of Krosno", "He was the fourth and youngest son of Henry VIII the Sparrow, Duke of Głogów"),
  edge("Henry VIII the Sparrow", "死亡地", "Szprotawa", missing="父亲正文列举统治地含 Szprotawa，但没有死亡地点关系；所有可见正文均未补足。")], ["labeled_body_lacks_death_place"]),
 "2wiki:1b911ab00bdb11eba7f7acde48001122": judgment(True, "母亲为 Christina of Saxony，其括注明确标出生地 b. Torgau。", [
  edge("Elizabeth of Denmark, Electress of Brandenburg", "母亲", "Christina of Saxony", "Elizabeth of Denmark, Electress of Brandenburg", "She was the daughter of King Hans of Denmark, Norway and Sweden and his spouse, Christina of Saxony."),
  edge("Christina of Saxony", "出生地", "Torgau", "Christina of Saxony", "Christina of Saxony (b. Torgau, 25 December 1461 – d. Odense, 8 December 1521)")]),
 "2wiki:7b3f24c80bdd11eba7f7acde48001122": judgment(True, "New York, New York 为参考 New York 的更明确表达，正文同样给出该死亡地。", [
  edge("Journal of a Crime", "导演", "William Keighley", "Journal of a Crime", "directed by William Keighley"),
  edge("William Keighley", "死亡地", "New York", "William Keighley", "William Jackson Keighley( August 4, 1889, Philadelphia, Pennsylvania – June 24, 1984, New York, New York)")], ["more_specific_place_equivalence"]),
 "2wiki:4b9f70180bda11eba7f7acde48001122": judgment(False, "1688 年份与完整日期不矛盾，但遗漏冻结答案的月日；按本次完整回答口径记为不完整，而非事实矛盾。", [
  edge("Antonio Lante Montefeltro della Rovere", "父亲", "Ippolito Lante Montefeltro della Rovere", "Antonio Lante Montefeltro della Rovere", "Lante was the son of Ippolito Lante Montefeltro della Rovere and Maria Cristina d'Altemps."),
  edge("Ippolito Lante Montefeltro della Rovere", "死亡日期", "29 June 1688", "Ippolito Lante Montefeltro della Rovere", "Ippolito Lante Montefeltro della Rovere (15 June 1618 – 29 June 1688)")], ["answer_date_precision_policy"]),
 "2wiki:8a77eae20bdd11eba7f7acde48001122": judgment(True, "父亲身份及28 January 1714死亡日期均在最终输入中。", [
  edge("Philip Stanhope, 3rd Earl of Chesterfield", "父亲", "Philip Stanhope, 2nd Earl of Chesterfield", "Philip Stanhope, 3rd Earl of Chesterfield", "the eldest son of Philip Stanhope, 2nd Earl of Chesterfield"),
  edge("Philip Stanhope, 2nd Earl of Chesterfield", "死亡日期", "28 January 1714", "Philip Stanhope, 2nd Earl of Chesterfield", "Philip Stanhope, 2nd Earl of Chesterfield PC FRS (1634 – 28 January 1714)")]),
 "2wiki:74f39a760bde11eba7f7acde48001122": judgment(False, "1608 是父亲 Montagu 的出生年，最终输入已给出其死亡日期25 July 1666。", [
  edge("Robert Bertie, 3rd Earl of Lindsey", "父亲", "Montagu Bertie, 2nd Earl of Lindsey", "Robert Bertie, 3rd Earl of Lindsey", "He was the son of Montagu Bertie, 2nd Earl of Lindsey and Martha Cokayne."),
  edge("Montagu Bertie, 2nd Earl of Lindsey", "死亡日期", "25 July 1666", "Montagu Bertie, 2nd Earl of Lindsey", "Montagu Bertie, 2nd Earl of Lindsey, KG, PC ( 1608 – 25 July 1666)")]),
 "2wiki:69af3b6c0bdd11eba7f7acde48001122": judgment(False, "Yoshino, Kyoto 混合两地，不能视为 Kyoto 的等价地名；正文只给南北朝驻地，没有父亲出生地。", [
  edge("Prince Kaneyoshi", "父亲", "Emperor Go-Daigo", "Prince Kaneyoshi", "The son of Emperor Go-Daigo"),
  edge("Emperor Go-Daigo", "出生地", "Kyoto", missing="Kyoto 与 Yoshino 在可见正文中是朝廷驻地，不是出生地；其他可见正文也未建立此出生关系。")], ["labeled_body_lacks_birth_place", "extra_place_not_equivalent"]),
 "2wiki:66772eb80bde11eba7f7acde48001122": judgment(True, "Palm Springs, California 与冻结参考一致，但正文只写退休到当地，未证明死于当地；答案正确性与证据充分性分开。", [
  edge("13 Fighting Men", "导演", "Harry W. Gerstad", "13 Fighting Men", "13 Fighting Men is a 1960 American drama film directed by Harry W. Gerstad"),
  edge("Harry W. Gerstad", "死亡地", "Palm Springs, California", missing="正文的 retired to Palm Springs 不是死亡地证据。")], ["labeled_body_lacks_death_place", "reference_match_without_relation_proof"]),
 "2wiki:f27e01300bdd11eba7f7acde48001122": judgment(True, "父亲为 Casimir I，正文 b. Kraków 明确标出生地。", [
  edge("Władysław I Herman", "父亲", "Casimir I the Restorer", "Władysław I Herman", "He was the second son of Casimir I the Restorer"),
  edge("Casimir I the Restorer", "出生地", "Kraków", "Casimir I the Restorer", "Casimir I the Restorer (b. Kraków, 25 July 1016 – d. Poznań, 28 November 1058)")]),
 "2wiki:a703e1160bdd11eba7f7acde48001122": judgment(False, "Macedon 为无关地名，未回答丈夫的死亡地；最终输入只取得妻子与丈夫绑定。", [
  edge("Ginevra d'Este", "丈夫", "Sigismondo Pandolfo Malatesta", "Ginevra d'Este", "She was the first of Sigismondo Pandolfo Malatesta's three wives."),
  edge("Sigismondo Pandolfo Malatesta", "死亡地", "Rimini", missing="丈夫正文未被检索返回；离线支持正文也只写 lord of Rimini，不能由统治地推死亡地。")], ["labeled_body_lacks_death_place"]),
 "2wiki:d94d1d720bdd11eba7f7acde48001122": judgment(None, "ASVEL Basket 的所有权及2011年效力关系有正文依据，但题目 work at 未给时间、冻结参考为 Spurs，不能无条件把另一时期雇主当等价答案。", [
  edge("Balance-toi", "表演者", "Tony Parker", "Balance-toi", "\"Balance-toi\" is a 2007 song recorded by NBA player Tony Parker."),
  edge("Tony Parker", "冻结链指定雇主", "San Antonio Spurs", "Tony Parker", "before joining the San Antonio Spurs of the National Basketball Association (NBA)")], ["ambiguous_employment_time", "documented_alternative_answer_requires_adjudication"]),
 "2wiki:df13ddf40bdd11eba7f7acde48001122": judgment(False, "Farīd Khān 是父亲原名，不是出生地点；Sasaram 在正文中只作为帝国首都出现。", [
  edge("Islam Shah Suri", "父亲", "Sher Shah Suri", "Islam Shah Suri", "he was the second son of Sher Shah Suri"),
  edge("Sher Shah Suri", "出生地", "Sasaram", missing="born Farīd Khān 表示原名，不表示出生地；capital in Sasaram 不能代替出生地。")], ["labeled_body_lacks_birth_place"]),
 "2wiki:015a72a60bde11eba7f7acde48001122": judgment(False, "Ornans 来自无关作品标题；父亲传记没有出生地点关系。", [
  edge("Bonne of Armagnac", "父亲", "Bernard VII, Count of Armagnac", "Bonne of Armagnac", "was the eldest daughter of Bernard VII, Count of Armagnac and Constable of France"),
  edge("Bernard VII, Count of Armagnac", "出生地", "Paris", missing="父亲正文描述 Paris 的入侵／被杀背景，不证明出生于 Paris；其他可见正文未补足。")], ["labeled_body_lacks_birth_place"]),
 "2wiki:2e92333a0bde11eba7f7acde48001122": judgment(None, "Bergen Museum 是正文明确记载的1915年任职机构，参考 University of Bergen 对应1917年；题目未给时间，不能把真实的不同机构直接判为同义或错误。", [
  edge("Eigil Helland-Hansen", "父亲", "Bjørn Helland-Hansen", "Eigil Helland-Hansen", "He was born in Årstad, Bergen to oceanographer Bjørn Helland-Hansen and Anna Marie Krag"),
  edge("Bjørn Helland-Hansen", "冻结链指定任职机构", "University of Bergen", "Bjørn Helland-Hansen", "in 1917 director of the Geophysical Institute, University of Bergen")], ["ambiguous_employment_time", "documented_alternative_answer_requires_adjudication"]),
 "2wiki:114313dc0bdc11eba7f7acde48001122": judgment(True, "NYU 是 Tisch 所属大学，正文明确写 New York University's Tisch School；问题只问在哪里学习，没有要求院系粒度，接受大学名称。", [
  edge("Home Fries", "导演", "Dean Parisot", "Home Fries", "Home Fries is a 1998 film directed by Dean Parisot"),
  edge("Dean Parisot", "就读学校", "Tisch / New York University", "Dean Parisot", "He graduated from New York University's Tisch School of the Arts.")], ["school_parent_university_granularity"]),
 "2wiki:d8c06f220bdc11eba7f7acde48001122": judgment(True, "Buenos Aires 与冻结参考一致，但正文中该城市只属于电影节，不证明导演死亡地。", [
  edge("Back Long Ago", "导演", "Ricardo Becher", "Back Long Ago", "Back Long Ago  is a 1969 Argentine film directed by Ricardo Becher"),
  edge("Ricardo Becher", "死亡地", "Buenos Aires", missing="Buenos Aires 在正文中只出现在电影节名称，未提供死亡地关系。")], ["labeled_body_lacks_death_place", "reference_match_without_relation_proof"]),
 "2wiki:92d51c3a0bda11eba7f7acde48001122": judgment(True, "配偶绑定及 executed ... in Pest 死亡地点均明确可见。", [
  edge("Antónia Zichy", "丈夫", "Lajos Batthyány", "Antónia Zichy", "wife of Lajos Batthyány"),
  edge("Lajos Batthyány", "死亡地", "Pest", "Lajos Batthyány", "was executed by firing squad in Pest on 6 October 1849")]),
 "2wiki:8796fb620bdb11eba7f7acde48001122": judgment(True, "St. Pauli, Hamburg 是正文明确给出的 Hamburg 内更具体出生地，接受为语义正确。", [
  edge("Peter the Mariner", "导演", "Reinhold Schünzel", "Peter the Mariner", "Peter the Mariner  is a 1929 German silent comedy drama film directed by Reinhold Schünzel"),
  edge("Reinhold Schünzel", "出生地", "Hamburg", "Reinhold Schünzel", "he was born in St. Pauli, the poorest part of Hamburg")], ["more_specific_place_equivalence"]),
 "musique:2hop__560534_91104": judgment(False, "重复查询结束，无最终答案；实际输入未取得 ATS-6。", [
  edge("IRS-P3", "任务执行／运营机构", "ISRO", "IRS-P3", "IRS-P3 was an experimental earth observation mission undertaken by ISRO."),
  edge("ISRO 通信卫星", "前身实验卫星", "ATS-6", missing="最终输入无 ATS-6 正文；离线 ATS-6 支持正文也未明确 forerunner 关系。")], ["labeled_body_lacks_forerunner_relation", "no_answer"]),
 "musique:2hop__468778_88165": judgment(True, "Keturah 与冻结参考一致，且 after the death of Sarah 限定可见；但 Hagar 正文只证明生育关系，未明确称配偶。", [
  edge("Hagar", "配偶", "Abraham", missing="Hagar 正文的 gave her to Abraham to bear a child／union 及母子关系不能单独证明正式 spouse 关系。"),
  edge("Abraham", "Sarah 死后所娶妻子", "Keturah", "Keturah", "According to the Book of Genesis, Abraham married Keturah after the death of his first wife, Sarah.")], ["spouse_vs_concubine_relation", "reference_match_with_incomplete_spouse_proof"]),
 "musique:2hop__442175_56873": judgment(False, "2015 是 Ward 后来加入 Forest 的年份，未回答 Forest 获 FA Cup 的1898和1959。", [
  edge("Jamie Ward", "所属球队", "Nottingham Forest", "Jamie Ward", "before joining Derby County in 2011 and later Nottingham Forest in 2015"),
  edge("Nottingham Forest", "FA Cup 获胜年份", "1898 and 1959", missing="最终输入无 Forest 获 FA Cup 的年份关系。")], ["multiple_historical_clubs"]),
}

PRIOR_REASONS = {
 "musique:2hop__864101_135373": "岛屿→Bering Sea→Vitus Bering 两边可见，答案正确。",
 "musique:2hop__478978_124498": "专辑→Matthieu Chedid→父亲 Louis Chedid 两边可见。",
 "musique:2hop__862994_69048": "出生地是 Delta State，答案人物的可见关系属于 Rivers State；目标辖区 chief judge 边缺失。",
 "musique:2hop__829358_754719": "出生城市 Pedro Leopoldo 位于 Minas Gerais 已可见，却选 Minsk Region。",
 "musique:2hop__789565_416401": "母子及父子关系均可见，却把王继鹏的表亲 Lady Li 当成目标子女。",
 "musique:2hop__506035_325512": "Smith→Poplarville→Pearl River County 两边可见。",
 "musique:2hop__337223_56873": "Forest 成员关系可见，FA Cup 获胜年份缺失；2015–16 为其他竞赛语境。",
 "musique:2hop__796221_18803": "第二次搜索取得 Shell 的2010年4月公告，回答与公告语义一致；严格 EM 未命中。",
 "musique:2hop__151750_141308": "专辑→Apple Records→Apple Corps Ltd 两边可见；Ltd 后缀不影响正确性。",
 "musique:2hop__418360_67370": "书籍作者 Bill Maher 及其节目2018年首播日期均可见。",
 "musique:2hop__96194_78303": "电影主演 Chuck Norris 及其 Walker 角色 Cordell Walker 均可见，却回答 Simon Moon。",
 "musique:2hop__703493_162341": "提名及最佳男配角类别已可见；78th Academy Awards 未回答必要的提名状态与类别。",
 "musique:2hop__158275_35739": "Windows 8 正文直接提供 Firefox 及其版本首次可用时间，单篇正文覆盖两边。",
 "musique:2hop__435087_55984": "歌曲正文提供 Brandi Carlile 表演及 Phil Hanseroth 创作关系。",
 "musique:2hop__109242_113442": "画作作者与作者生卒日期端点均可见，死亡日期正确。",
 "musique:2hop__69765_43945": "Common Sense 作者与推动脱离 Great Britain 独立的作用均可见。",
 "2wiki:982f8e440bdb11eba7f7acde48001122": "出版机构 Sigma Xi 及其1886年成立均可见。",
 "2wiki:67ecf3100bdc11eba7f7acde48001122": "导演身份与出生地 Poznan 均可见，接受地名拼写变体。",
 "2wiki:3a3d60020bdd11eba7f7acde48001122": "歌手 Anais 及来自 Dominican Republic 的关系可见；问题问来自何处，无须另证明出生地。",
 "2wiki:ef9d40a40bdb11eba7f7acde48001122": "导演 Habib Azar 与 American 国籍描述可见，接受美国国籍同义表达。",
 "2wiki:5dac1e8e0bdd11eba7f7acde48001122": "导演身份与 New York, New York 出生地点可见。",
 "2wiki:1dd33f080bde11eba7f7acde48001122": "父亲明确 born at Arnstadt，却回答其工作／死亡相关地 Eisenach。",
 "2wiki:764039340bde11eba7f7acde48001122": "父亲绑定及 Hyde Park, Westminster 死亡地点均可见。",
 "2wiki:7bda387a0bdb11eba7f7acde48001122": "父亲 Gerald Tanner 及24 August 1921出生日期均可见。",
 "2wiki:f299dd600bdd11eba7f7acde48001122": "父亲关系及1575死亡年均可见，AM 已回答1575。",
 "2wiki:49d965aa0bde11eba7f7acde48001122": "歌手 Yoyoy Villame 及 Las Pinas Medical Center 死亡地点可见；问题未限制只答城市。",
 "2wiki:ee3233920bda11eba7f7acde48001122": "父亲 Francesco I Crispo 及1397死亡年可见，没有混成 Francesco II。",
 "musique:2hop__498251_126089": "Laren 位于 North Holland 可见，主管人物边缺失；回答地区 Gooi 而非人物。",
 "musique:2hop__530579_13529": "父亲 Maradona 及 June 1982 签约 FC Barcelona 可见。",
}
