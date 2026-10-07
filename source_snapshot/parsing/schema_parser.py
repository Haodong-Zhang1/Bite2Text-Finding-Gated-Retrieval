#!/usr/bin/env python3
"""可复用的报告→schema 解析器(从 parse_reports.py 抽出, 无副作用, 供 oracle/eval 导入)。
v0 规则式; 最终版应加 LLM 复核。"""
import re

def hedged(t):
    return bool(re.search(r"slightly|appears?|seem|would appear|tendency|difficult|or slightly|almost|probably|nevertheless|not aligned|cannot (?:be )?(?:state|assess|determin|defin)", t, re.I))

def f_dentition(t):
    if re.search(r"early mixed dentition", t, re.I): return "early_mixed"
    if re.search(r"mixed dentition", t, re.I): return "mixed"
    return "permanent"

def f_constriction(t):
    if re.search(r"constriction of the maxilla|maxillary constriction|transverse constriction", t, re.I): return "present"
    if re.search(r"correct transverse|transverse relationships? .{0,30}within normal|inter-?arch relationships? appear.{0,20}within normal|on the transverse plane", t, re.I): return "absent"
    return None

def f_crossbite(t):
    if re.search(r"absence of crossbite|no (?:lateral )?crossbites?|without .{0,15}crossbite|no crossbites? are (?:present|observed)", t, re.I): return "absent"
    if re.search(r"almost-?present crossbite", t, re.I): return "borderline"
    if re.search(r"crossbite (?:of|on|involving|is|noted|present)|lateral crossbite|reverse bite|scissor", t, re.I): return "present"
    return None

def f_vertical(t):
    deep=bool(re.search(r"deep ?bite", t, re.I)); openb=bool(re.search(r"open ?bite", t, re.I))
    ob_inc=bool(re.search(r"overbite is increased|overbite .{0,20}increased|increased.{0,10}overbite", t, re.I))
    ob_norm=bool(re.search(r"overbite .{0,25}within normal", t, re.I))
    if openb: return "open_bite"
    if deep: return "deep_bite"
    if ob_inc: return "overbite_increased"
    if ob_norm or re.search(r"correct .{0,10}vertical|vertical .{0,20}within normal", t, re.I): return "normal"
    return None

def f_overjet(t):
    if re.search(r"negative overjet|reverse", t, re.I): return "negative"
    if re.search(r"overjet .{0,20}increased|increased .{0,10}overjet|markedly increased", t, re.I): return "increased"
    if re.search(r"overjet .{0,25}within normal", t, re.I): return "normal"
    return None

def f_midline(t):
    if re.search(r"midlines? .{0,20}(?:are )?centered|are coincident|centered with each other", t, re.I) and not re.search(r"not coincident|non-coincident", t, re.I):
        return "centered"
    if re.search(r"not coincident|non-?coincident|deviated|not centered|do not coincide", t, re.I): return "deviated"
    if re.search(r"midlines? .{0,30}difficult", t, re.I): return "unassessable"
    return None

def _curve(t, which):
    seg=t.lower(); idx=seg.find(which)
    if idx<0: return None
    window=seg[max(0,idx-40):idx+60]
    if re.search(r"increased|accentuated", window): return "increased"
    if re.search(r"within normal|normal limits", window): return "normal"
    if re.search(r"cur[vy]es? of spee and .{0,20}wilson[^.]*(increased|accentuated)", seg): return "increased"
    if re.search(r"cur[vy]es? of spee and .{0,20}wilson[^.]*within normal", seg): return "normal"
    return None
def f_spee(t): return _curve(t,"spee")
def f_wilson(t): return _curve(t,"wilson")

def _crowd_level(s):
    s=s.lower()
    if "mild to moderate" in s or "mild-to-moderate" in s: return "mild_moderate"
    if "moderate to severe" in s or "moderate-to-severe" in s: return "moderate_severe"
    if "severe" in s: return "severe"
    if "moderate" in s: return "moderate"
    if "mild" in s: return "mild"
    return None

def f_crowding(t):
    up=lo=None; tl=t.lower()
    if re.search(r"no crowding|crowding is absent|both .{0,10}aligned|without crowding", tl):
        up=up or "none"; lo=lo or "none"
    for m in re.finditer(r"(mild(?:[ -]to[ -]moderate)?|moderate(?:[ -]to[ -]severe)?|severe|moderate) crowding[^.]*?(upper|maxillary|lower|mandibular)", tl):
        lvl=_crowd_level(m.group(1)); up,lo=((lvl,lo) if m.group(2) in ("upper","maxillary") else (up,lvl))
    for m in re.finditer(r"(upper|maxillary|lower|mandibular) arch[^.]*?(mild(?:[ -]to[ -]moderate)?|moderate(?:[ -]to[ -]severe)?|severe|moderate) crowding", tl):
        lvl=_crowd_level(m.group(2)); up,lo=((lvl,lo) if m.group(1) in ("upper","maxillary") else (up,lvl))
    m=re.search(r"(mild(?:[ -]to[ -]moderate)?|moderate(?:[ -]to[ -]severe)?|severe|moderate) crowding .{0,40}(?:upper and lower|both .{0,6}arch|lower and upper)", tl)
    if m: lvl=_crowd_level(m.group(1)); up=up or lvl; lo=lo or lvl
    return up, lo

def _class_in(seg):
    if re.search(r"cannot (?:be )?(?:assess|determin|defin)|not assessable|not be defined|not be assessed", seg, re.I): return "NA"
    if re.search(r"class iii|class 3", seg, re.I): return "III"
    if re.search(r"class ii\b|class 2\b", seg, re.I): return "II"
    if re.search(r"class i\b|class 1\b", seg, re.I): return "I"
    return None

def _relation_class(segment, relation):
    relation_pattern=rf"\b{relation}\b"
    shared=bool(re.search(r"molar and canine|canine and molar",segment,re.I))
    if shared:
        return _class_in(segment)
    patterns=[
        rf"(?:class\s*(?:iii|ii|i|3|2|1)|not assessable|cannot be assessed).{{0,35}}{relation_pattern}",
        rf"{relation_pattern}.{{0,35}}(?:class\s*(?:iii|ii|i|3|2|1)|not assessable|cannot be assessed)",
    ]
    for pattern in patterns:
        match=re.search(pattern,segment,re.I)
        if match:
            return _class_in(match.group(0))
    return None

def f_sagittal(t):
    res={"molar_R":None,"molar_L":None,"canine_R":None,"canine_L":None}
    def fill(segment, sides):
        molar=_relation_class(segment,"molar")
        canine=_relation_class(segment,"canine")
        for side in sides:
            if molar is not None: res[f"molar_{side}"]=molar
            if canine is not None: res[f"canine_{side}"]=canine

    for sentence in re.split(r"(?<=[.!?])\s+",t):
        markers=list(re.finditer(r"\b(?:on|to)\s+the\s+(right|left)\b|\b(right|left)(?:-hand)?\s+side\b",sentence,re.I))
        if markers:
            for i,marker in enumerate(markers):
                side=(marker.group(1) or marker.group(2)).upper()[0]
                end=markers[i+1].start() if i+1<len(markers) else len(sentence)
                fill(sentence[marker.start():end],[side])
        elif re.search(r"bilateral|both .{0,10}sides|right and left",sentence,re.I):
            fill(sentence,["R","L"])

        # Relation-first form: "molar relationship is Class II on the right
        # and Class I on the left". Associate each side with its nearest class
        # mention, then with the nearest relation name in the same sentence.
        side_mentions=list(re.finditer(r"\b(right|left)\b",sentence,re.I))
        class_mentions=list(re.finditer(
            r"class\s*(?:iii|ii|i|3|2|1)|not assessable|cannot be assessed",
            sentence,
            re.I,
        ))
        relation_mentions=list(re.finditer(r"\b(molar|canine)\b",sentence,re.I))
        shared=bool(re.search(r"molar and canine|canine and molar",sentence,re.I))

        class_side_pairs=[]
        for match in re.finditer(
            r"(class\s*(?:iii|ii|i|3|2|1)|not assessable|cannot be assessed)"
            r".{0,12}?\b(?:on|to)\s+the\s+(right|left)\b",
            sentence,
            re.I,
        ):
            class_side_pairs.append((match.group(2), _class_in(match.group(1))))
        for match in re.finditer(
            r"\b(right|left)(?:-hand)?\s+side\b.{0,12}?"
            r"(class\s*(?:iii|ii|i|3|2|1)|not assessable|cannot be assessed)",
            sentence,
            re.I,
        ):
            class_side_pairs.append((match.group(1), _class_in(match.group(2))))
        for side_text,value in class_side_pairs:
            if not relation_mentions:
                continue
            side="R" if side_text.lower()=="right" else "L"
            targets=("molar","canine") if shared else (
                min(
                    relation_mentions,
                    key=lambda match: abs(match.start()-sentence.lower().find(side_text.lower())),
                ).group(1).lower(),
            )
            for relation in targets:
                res[f"{relation}_{side}"]=value

        for side_match in side_mentions:
            if not class_mentions or not relation_mentions:
                continue
            side="R" if side_match.group(1).lower()=="right" else "L"
            class_match=min(
                class_mentions,
                key=lambda match: abs(match.start()-side_match.start()),
            )
            value=_class_in(class_match.group(0))
            nearest_relation=min(
                relation_mentions,
                key=lambda match: abs(match.start()-side_match.start()),
            ).group(1).lower()
            targets=("molar","canine") if shared else (nearest_relation,)
            for relation in targets:
                key=f"{relation}_{side}"
                if res[key] is None:
                    res[key]=value
    return res

def f_missing(t):
    teeth=set()
    for m in re.finditer(r"(?:absence of|absent|not present|missing|are absent)[^.]{0,40}?(\d{2}(?:[-–]\d{2})*)",t,re.I):
        teeth.add(m.group(1))
    for m in re.finditer(r"(?:tooth|teeth) (\d{2}(?:[-–]\d{2})*)[^.]{0,25}?(?:not present|absent|missing)",t,re.I):
        teeth.add(m.group(1))
    return ";".join(sorted(teeth)) if teeth else None

FIELDS=["dentition","constriction","crossbite","vertical","overjet","midline","spee","wilson",
        "crowding_upper","crowding_lower","molar_R","molar_L","canine_R","canine_L"]

def parse_report(t):
    row={"dentition":f_dentition(t),"constriction":f_constriction(t),"crossbite":f_crossbite(t),
         "vertical":f_vertical(t),"overjet":f_overjet(t),"midline":f_midline(t),
         "spee":f_spee(t),"wilson":f_wilson(t)}
    up,lo=f_crowding(t); row["crowding_upper"]=up; row["crowding_lower"]=lo
    row.update(f_sagittal(t))
    row["missing_teeth"]=f_missing(t)
    row["hedged"]=hedged(t)
    row["has_NA_field"]=any(row[k]=="NA" for k in ("molar_R","molar_L","canine_R","canine_L"))
    row["n_words"]=len(t.split())
    return row

# 评分用归一化: 合并临床同义类(D2 发现)
def normalize_for_scoring(schema, merge_vertical=True):
    s=dict(schema)
    if merge_vertical and s.get("vertical") in ("deep_bite","overbite_increased"):
        s["vertical"]="increased_overlap"
    return s
