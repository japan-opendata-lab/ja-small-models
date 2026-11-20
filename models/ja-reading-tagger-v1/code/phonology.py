#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 用 日本語表記・音韻ユーティリティ。

- ひらがな⇔カタカナ正規化
- 漢字判定・書記片抽出
- 送り仮名・接頭辞/接尾辞かなの剥離 (strip_okurigana)
- 表記読み（ルビ用カタカナ）から発音読み（TTS用・UniDic準拠カタカナ）への変換
"""
import re
from typing import Dict, List, Optional, Tuple

KANJI_CHARS = "々ヶヵ〆〇"
KANJI_RE = re.compile(r"[一-龥々ヶヵ〆〇]")
PERSON_EXCLUDE_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]")

_O_ROW = frozenset("オコソトノホモヨロヲゴゾドボポォョ")
_E_ROW = frozenset("エケセテネヘメエレゲゼデベペェ")
_U_ROW = frozenset("ウクスツヌフムユルグズヅブプゥュヴ")


def is_kanji_char(ch: str) -> bool:
    """1文字が対象漢字（CJK統合漢字および々ヶヵ〆〇）か判定する。"""
    if not ch:
        return False
    return ("\u4e00" <= ch <= "\u9fff") or (ch in KANJI_CHARS)


def has_kanji(text: str) -> bool:
    """文字列に1文字以上の対象漢字が含まれるか判定する。"""
    return bool(KANJI_RE.search(text))


def is_all_kanji(text: str) -> bool:
    """文字列が空でなく全文字が対象漢字か判定する。"""
    return bool(text) and all(is_kanji_char(c) for c in text)


def hira_to_kata(text: str) -> str:
    """ひらがなをカタカナに変換する（長音・記号はそのまま保持）。"""
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        elif ch == "\u309d":  # ゝ
            out.append("\u30fd")
        elif ch == "\u309e":  # ゞ
            out.append("\u30fe")
        else:
            out.append(ch)
    return "".join(out)


def kata_to_hira(text: str) -> str:
    """カタカナをひらがなに変換する（ヴは保持またはう゛ではなくそのまま、長音ーも保持）。"""
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        elif ch == "\u30fd":  # ヽ
            out.append("\u309d")
        elif ch == "\u30fe":  # ヾ
            out.append("\u309e")
        else:
            out.append(ch)
    return "".join(out)


def normalize_kana(text: str) -> str:
    """読みを標準の全角カタカナ（ひらがな→カタカナ、空白除去）へ正規化する。"""
    t = text.strip().replace(" ", "").replace("\u3000", "")
    # 歴史的・特殊仮名表記のうち「ゐ」「ゑ」などはそのまま残すが基本はhira_to_kata
    return hira_to_kata(t)


def is_valid_reading_str(text: str) -> bool:
    """読み文字列が正当な仮名列（カタカナ/ひらがな/ー）のみで構成されるか判定する。"""
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        if not (
            (0x3041 <= code <= 0x3096)
            or (0x30A1 <= code <= 0x30FA)
            or ch in ("ー", "・")
        ):
            return False
    return True


def _kana_variants(kana_kata: str) -> List[str]:
    """仮名セグメントの検索候補（小書き「ヶ/ヵ」が「カ/ガ/ケ」と読まれる場合等を除く標準列）。"""
    return [kana_kata]


def strip_okurigana(
    surface: str, reading_kata: str
) -> Optional[List[Tuple[int, int, str, str]]]:
    """表層文字列と全体の読みカタカナから、漢字ブロック単位の (start, end, 漢字列, 読みカタカナ) を抽出する。

    例:
      strip_okurigana("開く", "ヒラク") -> [(0, 1, "開", "ヒラ")]
      strip_okurigana("行った", "イッタ") -> [(0, 1, "行", "イ")]
      strip_okurigana("行った", "オコナッタ") -> [(0, 1, "行", "オコナ")]
      strip_okurigana("読み上げた", "ヨミアゲタ") -> [(0, 1, "読", "ヨ"), (2, 3, "上", "ア")]
      strip_okurigana("お父さん", "オトウサン") -> [(1, 2, "父", "トウ")]
      strip_okurigana("人気", "ニンキ") -> [(0, 2, "人気", "ニンキ")]
    """
    if not surface or not reading_kata:
        return None
    r_kata = normalize_kana(reading_kata)
    if not has_kanji(surface):
        return []

    # 全体を連続する漢字セグメント / 非漢字(かな)セグメントに分割
    # ただし「ヶ」「ヵ」が単独や数詞後で「ガ/カ」と読まれる場合があるため、
    # 「一ヶ月」「霞ヶ関」のように前後を漢字に挟まれたヶ/ヵは漢字塊に含まれている。
    segs: List[Tuple[bool, int, int, str]] = []
    n = len(surface)
    i = 0
    while i < n:
        flag = is_kanji_char(surface[i])
        j = i + 1
        while j < n and is_kanji_char(surface[j]) == flag:
            j += 1
        segs.append((flag, i, j, surface[i:j]))
        i = j

    # 全漢字ならそのまま返す
    if len(segs) == 1 and segs[0][0]:
        return [(0, n, surface, r_kata)]

    # かなセグメントはカタカナへ
    seg_info: List[Tuple[bool, int, int, str, str]] = []
    for flag, a, b, txt in segs:
        k = "" if flag else normalize_kana(txt)
        seg_info.append((flag, a, b, txt, k))

    # 再帰アラインメント（左から一致位置を探索、末尾かなは後ろから確定させるため全探索して最短・自然解を選ぶ）
    r_len = len(r_kata)
    solutions: List[List[Tuple[int, int, str, str]]] = []

    def dfs(s_idx: int, r_pos: int, acc: List[Tuple[int, int, str, str]]) -> None:
        if s_idx == len(seg_info):
            if r_pos == r_len:
                solutions.append(list(acc))
            return

        is_k, a, b, txt, k_pat = seg_info[s_idx]
        if not is_k:
            if not k_pat:
                dfs(s_idx + 1, r_pos, acc)
                return
            if r_kata.startswith(k_pat, r_pos):
                dfs(s_idx + 1, r_pos + len(k_pat), acc)
            return

        # 漢字セグメント: 次の非漢字セグメント（存在すれば）まで伸ばす
        min_kanji_read = max(1, (b - a))
        # 稀に2字熟字訓や短縮音で文字数より短い読みは起きうるか? 例: 大人(おとな=3>2), 今日(きょう=3>2)
        # ただし「一(いち)」や「行(い)」は1文字。熟語2文字で1モーラは無いため min_kanji_read = 1 が安全
        min_len = 1
        if s_idx == len(seg_info) - 1:
            # 最後のセグメントが漢字なら残り全部を割り当てる
            if r_len - r_pos >= min_len:
                acc.append((a, b, txt, r_kata[r_pos:]))
                dfs(s_idx + 1, r_len, acc)
                acc.pop()
            return

        # 次は必ずかなセグメント
        next_k_pat = seg_info[s_idx + 1][4]
        if not next_k_pat:
            return
        # 次以降の残りかな長合計を下界として探索範囲を絞る
        rem_kana_len = sum(len(x[4]) for x in seg_info[s_idx + 1 :] if not x[0])
        rem_kanji_cnt = sum(1 for x in seg_info[s_idx + 1 :] if x[0])
        max_end = r_len - rem_kana_len - rem_kanji_cnt
        for end_pos in range(r_pos + min_len, max_end + 1):
            if r_kata.startswith(next_k_pat, end_pos):
                acc.append((a, b, txt, r_kata[r_pos:end_pos]))
                dfs(s_idx + 1, end_pos, acc)
                acc.pop()

    dfs(0, 0, [])
    if not solutions:
        return None

    # 複数解がある場合（例: 「見えた」ではないが「戦った(タタカッタ)」でタが複数出るケースなど）、
    # 漢字セグメント長が文字数に近い、または末尾かなが最右一致する解（最後の解が最長語幹/最右一致傾向）を選ぶ。
    # 例えば「はらはら」系では各漢字1〜3文字程度なので、最も均等かつ最後の送り仮名を長く取りすぎない最初の自然解より、
    # 「戦った」=「戦(タタカ)+った(ッタ)」のように、途中の送り仮名一致は短い最初の一致、末尾は最右一致がよい。
    # 具体的には: 最後のセグメントがかなの場合、最も末尾側でマッチする解（=漢字語幹が最長ではなく正しい語幹、例: 「戦った」は末尾1つしか「ッタ」がない。
    # もし「行った(オコナッタ)」でかなが「った」なら末尾1箇所のみ。「見えた」=ミエ+タ は1箇所。「支えた」=ササエ+タ も1箇所。
    # 「言い合い」= イ+イ+ア+イ (かな: い, あ, い): 「イ」が複数ある場合は最初の「イ」を1文字目の「言」に当てるのが自然！
    # したがって最小の文字数差になる解をスコアで選ぶ:
    if len(solutions) == 1:
        return solutions[0]

    def score_sol(sol: List[Tuple[int, int, str, str]]) -> Tuple[float, int]:
        # 各漢字ブロックの理想長は ~1.8 * kanji_len とし、各ブロックが1文字以上均等になる解を優先
        penalty = 0.0
        for a, b, _, rk in sol:
            klen = b - a
            ratio = len(rk) / max(1, klen)
            penalty += abs(ratio - 1.8)
            if len(rk) == 0:
                penalty += 100.0
        return (penalty, -len(sol[0][3]))

    solutions.sort(key=score_sol)
    return solutions[0]


def rule_kana_to_pron(reading_kata: str) -> str:
    """表記読みカタカナを UniDic 準拠の発音読みカタカナへ規則変換する。

    主に漢字スパン（語幹・漢語・和語名詞等）を対象とし、
    1) 四つ仮名正規化（ヂ→ジ, ヅ→ズ）
    2) 長音化（オ段+ウ/オ → オ段+ー, エ段+イ → エ段+ー, ウ段+ウ → ウ段+ー）
    を適用する。
    """
    k = normalize_kana(reading_kata).replace("ヂ", "ジ").replace("ヅ", "ズ")
    out: List[str] = []
    i = 0
    n = len(k)
    while i < n:
        ch = k[i]
        if i + 1 < n:
            nxt = k[i + 1]
            if ch in _O_ROW and (nxt == "ウ" or nxt == "オ"):
                out.append(ch)
                out.append("ー")
                i += 2
                continue
            if ch in _E_ROW and nxt == "イ":
                out.append(ch)
                out.append("ー")
                i += 2
                continue
            if ch in _U_ROW and nxt == "ウ":
                out.append(ch)
                out.append("ー")
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def kana_to_pron(
    reading_kata: str,
    surface: str = "",
    pron_map: Optional[Dict[str, str]] = None,
) -> str:
    """統合辞書発音マップを最優先し、未登録語には規則変換を適用して発音読みカタカナを返す。"""
    k = normalize_kana(reading_kata)
    if pron_map:
        if surface:
            sk_key = f"{surface}\t{k}"
            if sk_key in pron_map:
                return pron_map[sk_key]
        if k in pron_map:
            return pron_map[k]
    return rule_kana_to_pron(k)
