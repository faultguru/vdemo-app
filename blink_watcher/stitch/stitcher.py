"""Event stitching: groups a day's related clips into activities, by fixed rules (no AI).

    python -m blink_watcher stitch                     # today: the activities of 2+ clips, their links and scores
    python -m blink_watcher stitch --date 2026-10-17   # a past day (tune STITCH_* settings on it)
    python -m blink_watcher stitch --all               # also the single clips

Each clip gets a time window — its activity, the processed (trimmed) part of the
recording, from the clip's .metadata.json; the whole recording, or
STITCH_DEFAULT_CLIP_SEC, when that is unknown — and a signature from its summary:
people broken into attributes (colour+garment pairs, hair, type, carried items;
word lists in stitch/vocab.json, or STITCH_VOCAB_FILE), labels / actions / types / vehicles, and the
description's words. Two clips close in time are linked when one rule holds:

  overlap       other camera, windows overlap (within OVERLAP_SLACK_SEC), score >= MIN_SCORE_OVERLAP
  continuation  same camera, next clip within CONTINUATION_GAP_SEC,        score >= MIN_SCORE_CONTINUATION
  transition    other camera, starting within TRANSITION_GAP_SEC after,    score >= MIN_SCORE_TRANSITION
                (less between CAMERA_NEIGHBORS, more between non-neighbours when that map is set)
  track         same subject per the events' track field, within TRACK_GAP_SEC

Clips saved without analysis (security mode HOME, indoor cameras: "fast") have
no summary to compare, so they link by time alone and only with each other:
continuation on the same camera, overlap on another.

score = 0.6 people + 0.25 labels/actions/types/vehicles + 0.15 description words
(parts both clips lack are left out). Same garment in clearly different colours,
or people counts two or more apart, means no link. Overlap and transition need
people (or vehicles) on both sides. Linked clips form an activity; one longer
than MAX_SPAN_SEC is cut at its largest gap. Nothing is stored: activities are
derived on every call, so a settings change regroups every day.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date as Date, datetime, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from blink_watcher.chat import store as chat_store
from blink_watcher.core.logs import setup_logging
from blink_watcher.settings import Settings, load_settings
from blink_watcher.settings.cameras import camera_key

log = logging.getLogger("blink_watcher.stitch.stitcher")

PROGRAM = "stitch"
UNKNOWN_CAMERA = "(unknown camera)"
W_PEOPLE, W_CONTEXT, W_WORDS = 0.6, 0.25, 0.15
SPLIT = re.compile(r"[,;/()]|\band\b|\bwith\b|\bover\b|\bunder\b|\bcarrying\b|\bholding\b")
WORD = re.compile(r"[a-z]+(?:-[a-z]+)?")


# ---------- vocabulary ----------

@dataclass
class Vocab:
    colours: dict[str, str]
    compat: dict[str, set[str]]
    garments: dict[str, str]
    hair: dict[str, str]
    people: dict[str, str]
    attributes: dict[str, str]
    objects: dict[str, str]
    vehicles: dict[str, str]
    stopwords: set[str]

    @classmethod
    def load(cls, path: Optional[Path]) -> "Vocab":
        """`path`, or the built-in vocab.json (package data)."""
        source = Path(path) if path else files("blink_watcher.stitch") / "vocab.json"
        d = json.loads(source.read_text(encoding="utf-8"))
        compat: dict[str, set[str]] = {}
        for a, bs in (d.get("colour_compat") or {}).items():  # symmetric
            for b in bs:
                compat.setdefault(a, set()).add(b)
                compat.setdefault(b, set()).add(a)
        return cls(d["colours"], compat, d["garments"], d["hair"], d["people"], d["attributes"],
                   d["objects"], d["vehicles"], set(d.get("stopwords") or []))

    def colours_match(self, a: str, b: str) -> bool:
        return a == b or b in self.compat.get(a, ())


# ---------- signatures ----------

@dataclass
class Person:
    tokens: set[str] = field(default_factory=set)
    garments: dict[str, set[str]] = field(default_factory=dict)  # garment -> colours


def _person(p: dict, v: Vocab) -> Person:
    out = Person()
    text = " ".join(str(p.get(k) or "") for k in ("appearance", "appearance_details")).lower()
    for phrase in SPLIT.split(text):
        words = WORD.findall(phrase)
        if "hair" in words:
            out.tokens |= {f"hair:{v.hair[w]}" for w in words if w in v.hair}
            out.tokens |= {f"hair:{v.colours[w]}" for w in words if w in v.colours}
            continue
        garments = [v.garments[w] for w in words if w in v.garments]
        colours = [v.colours[w] for w in words if w in v.colours]
        for g in garments:
            out.tokens.add(f"g:{g}")
            out.tokens |= {f"{c}:{g}" for c in colours}
            out.garments.setdefault(g, set()).update(colours)
        out.tokens |= {f"p:{v.people[w]}" for w in words if w in v.people}
        out.tokens |= {f"a:{v.attributes[w]}" for w in words if w in v.attributes}
        out.tokens |= {f"o:{v.objects[w]}" for w in words if w in v.objects}
    if p.get("type"):
        out.tokens.add(f"p:{v.people.get(str(p['type']).lower(), str(p['type']).lower())}")
    for o in p.get("objects") or []:
        for w in WORD.findall(str(o).lower()):
            if w in v.objects:
                out.tokens.add(f"o:{v.objects[w]}")
    return out


def _stem(w: str) -> str:
    for suffix, keep in (("ing", 4), ("ed", 4), ("s", 4)):
        if w.endswith(suffix) and len(w) - len(suffix) >= keep:
            return w[: -len(suffix)]
    return w


@dataclass
class Signature:
    people: list[Person]
    vehicles: int
    context: set[str]
    words: set[str]


def signature(summary: dict, v: Vocab) -> Signature:
    people = [_person(p, v) for p in summary.get("people") or [] if isinstance(p, dict)]
    context = {f"l:{x}" for x in summary.get("labels") or []}
    context |= {f"x:{x}" for x in summary.get("actions") or []}
    context |= {f"t:{x}" for x in summary.get("people_types") or []}
    vehicles = [x for x in summary.get("vehicles") or [] if isinstance(x, dict)]
    for veh in vehicles:
        for w in WORD.findall(" ".join(str(veh.get(k) or "") for k in ("appearance", "appearance_details")).lower()):
            if w in v.vehicles:
                context.add(f"v:{v.vehicles[w]}")
            elif w in v.colours:
                context.add(f"vc:{v.colours[w]}")
    words = {_stem(w) for w in WORD.findall(str(summary.get("description") or "").lower())
             if len(w) > 2 and w not in v.stopwords}
    return Signature(people, len(vehicles), context, words)


# ---------- similarity ----------

def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _conflict(a: Person, b: Person, v: Vocab) -> bool:
    """The same garment in clearly different colours."""
    for g in a.garments.keys() & b.garments.keys():
        ca, cb = a.garments[g], b.garments[g]
        if ca and cb and not any(v.colours_match(x, y) for x in ca for y in cb):
            return True
    return False


def _people_score(a: list[Person], b: list[Person], v: Vocab) -> float:
    """Average best match of the smaller group's people (each used once)."""
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    free, total = list(big), 0.0
    for p in small:
        scored = [(0.0 if _conflict(p, q, v) else _jaccard(p.tokens, q.tokens), i) for i, q in enumerate(free)]
        best, i = max(scored, default=(0.0, -1))
        total += best
        if i >= 0:
            free.pop(i)
    return total / len(small)


def similarity(a: Signature, b: Signature, v: Vocab) -> float:
    if a.people and b.people and abs(len(a.people) - len(b.people)) >= 2:
        return 0.0
    parts: list[tuple[float, float]] = []
    if a.people or b.people:
        parts.append((W_PEOPLE, _people_score(a.people, b.people, v) if a.people and b.people else 0.0))
    if a.context or b.context:
        parts.append((W_CONTEXT, _jaccard(a.context, b.context)))
    if a.words or b.words:
        parts.append((W_WORDS, _jaccard(a.words, b.words)))
    if not parts:
        return 0.0
    score = sum(w * x for w, x in parts) / sum(w for w, _ in parts)
    if a.people and b.people and len(a.people) != len(b.people):
        score *= 0.85
    return score


# ---------- clips ----------

@dataclass
class Clip:
    id: str
    camera: str
    start: float          # epoch seconds: the activity window (= the processed video)
    end: float
    window: str           # trim | original | default
    sig: Signature
    track: set[str]       # event ids naming the same subject (track.id, same_as)
    day: str
    analysed: bool = True  # False: saved without analysis (no summary to compare)


def camera_of(event: dict, meta: dict) -> str:
    """The camera's name, as recorded with the event (or in its .metadata.json)."""
    return event.get("camera") or meta.get("camera") or UNKNOWN_CAMERA


def _window(recorded: float, meta: dict, default_sec: float) -> tuple[float, float, str]:
    trim = next((p for p in meta.get("processors") or [] if p.get("name") == "trim"), None)
    if trim and trim.get("duration_sec"):
        head = float(trim.get("removed_head_sec") or trim.get("offset_sec") or 0.0)
        return recorded + head, recorded + head + float(trim["duration_sec"]), "trim"
    length = (meta.get("original") or {}).get("duration_sec") or (meta.get("info") or {}).get("duration_sec")
    if length:
        return recorded, recorded + float(length), "original"
    return recorded, recorded + default_sec, "default"


class Stitcher:
    """Groups events into activities. Keeps the clips' metadata facts cached by
    file mtime, so polling a day again only re-reads changed files."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.root = settings.storage_path
        self.chat_dir = settings.chat_dir
        self.tz = ZoneInfo(settings.timezone)
        self.vocab = Vocab.load(settings.stitch_vocab_file)
        self._meta: dict[str, tuple[float, dict]] = {}
        nb = settings.stitch_camera_neighbors
        self.neighbors = {(camera_key(a), camera_key(b)) for a, bs in nb.items() for b in bs}
        self.neighbors |= {(b, a) for a, b in self.neighbors}

    def _metadata(self, rel: Optional[str]) -> dict:
        if not rel:
            return {}
        path = self.root / rel
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return {}
        hit = self._meta.get(rel)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        self._meta[rel] = (mtime, data)
        return data

    def clip(self, event: dict, day: str) -> Optional[Clip]:
        try:
            recorded = datetime.fromisoformat(event["recorded_at"]).timestamp()
        except (KeyError, TypeError, ValueError):
            return None
        meta = self._metadata(event.get("metadata"))
        start, end, window = _window(recorded, meta, self.s.stitch_default_clip_sec)
        tr = event.get("track") or {}
        track = set(tr.get("same_as") or []) | ({tr["id"]} if tr.get("id") else set())
        return Clip(event["event_id"], camera_of(event, meta), start, end, window,
                    signature(event.get("summary") or {}, self.vocab), track, day, analysed=not event.get("fast"))

    # ---------- links ----------

    def link(self, a: Clip, b: Clip) -> Optional[dict]:
        """The rule linking a and b (a starts first), or None."""
        s = self.s
        gap = b.start - a.end                       # < 0: the windows overlap
        same_cam = camera_key(a.camera) == camera_key(b.camera)
        if not (a.analysed and b.analysed):
            return self._link_unanalysed(a, b, gap, same_cam)
        subject = (a.sig.people and b.sig.people) or (a.sig.vehicles and b.sig.vehicles)
        score = similarity(a.sig, b.sig, self.vocab)
        out = {"a": a.id, "b": b.id, "score": round(score, 2), "gap_sec": round(gap, 1)}
        if (a.id in b.track or b.id in a.track or (a.track & b.track)) and gap <= s.stitch_track_gap_sec:
            return {**out, "kind": "track",
                    "reason": f"same subject (track) · {self._gap_text(gap)}"}
        if same_cam:
            if gap <= s.stitch_continuation_gap_sec and score >= s.stitch_min_score_continuation:
                return {**out, "kind": "continuation",
                        "reason": f"continued on the same camera · {self._gap_text(gap)} · {score:.0%} match"}
            return None
        if not subject:
            return None
        if gap <= s.stitch_overlap_slack_sec and score >= s.stitch_min_score_overlap:
            return {**out, "kind": "overlap",
                    "reason": f"seen together · {self._gap_text(gap)} · {score:.0%} match"}
        if 0 < gap <= s.stitch_transition_gap_sec:
            need = s.stitch_min_score_transition
            if self.neighbors:
                pair = (camera_key(a.camera), camera_key(b.camera))
                need += -s.stitch_neighbor_bonus if pair in self.neighbors else s.stitch_non_neighbor_penalty
            if score >= need:
                return {**out, "kind": "transition",
                        "reason": f"moved {a.camera} → {b.camera} · {gap:.0f}s later · {score:.0%} match"}
        return None

    def _link_unanalysed(self, a: Clip, b: Clip, gap: float, same_cam: bool) -> Optional[dict]:
        """Clips saved without analysis: by time alone, both unanalysed."""
        if a.analysed or b.analysed:
            return None
        out = {"a": a.id, "b": b.id, "score": None, "gap_sec": round(gap, 1)}
        if same_cam and gap <= self.s.stitch_continuation_gap_sec:
            return {**out, "kind": "continuation",
                    "reason": f"continued on the same camera · {self._gap_text(gap)} · not analysed"}
        if not same_cam and gap <= self.s.stitch_overlap_slack_sec:
            return {**out, "kind": "overlap", "reason": f"recorded together · {self._gap_text(gap)} · not analysed"}
        return None

    @staticmethod
    def _gap_text(gap: float) -> str:
        if gap < 0:
            return f"overlap {-gap:.0f}s"
        return f"{gap:.0f}s later" if gap < 120 else f"{gap / 60:.0f} min later"

    # ---------- grouping ----------

    def group(self, clips: list[Clip]) -> list[tuple[list[Clip], list[dict]]]:
        clips = sorted(clips, key=lambda c: (c.start, c.id))
        reach = max(self.s.stitch_overlap_slack_sec, self.s.stitch_continuation_gap_sec,
                    self.s.stitch_transition_gap_sec, self.s.stitch_track_gap_sec)
        links: list[dict] = []
        for i, a in enumerate(clips):
            for b in clips[i + 1:]:
                if b.start - a.end > reach:  # starts only grow from here
                    break
                ln = self.link(a, b)
                if ln:
                    links.append(ln)
        return self._components(clips, links)

    def _components(self, clips: list[Clip], links: list[dict]) -> list[tuple[list[Clip], list[dict]]]:
        ids = {c.id for c in clips}
        parent = {c.id: c.id for c in clips}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        inside = [ln for ln in links if ln["a"] in ids and ln["b"] in ids]
        for ln in inside:
            parent[find(ln["a"])] = find(ln["b"])
        comps: dict[str, list[Clip]] = {}
        for c in clips:
            comps.setdefault(find(c.id), []).append(c)
        out = []
        for members in comps.values():
            members.sort(key=lambda c: (c.start, c.id))
            span = max(c.end for c in members) - members[0].start
            if len(members) > 1 and span > self.s.stitch_max_span_sec:
                # cut at the largest gap, then regroup each side on its own links
                reach, best, at = members[0].end, None, 1
                for k in range(1, len(members)):
                    gap = members[k].start - reach
                    if best is None or gap > best:
                        best, at = gap, k
                    reach = max(reach, members[k].end)
                out += self._components(members[:at], inside)
                out += self._components(members[at:], inside)
            else:
                mids = {c.id for c in members}
                out.append((members, [ln for ln in inside if ln["a"] in mids]))
        return out

    # ---------- a day ----------

    def day(self, day: str, events: Optional[list[dict]] = None) -> list[dict]:
        """The day's activities (singletons included), newest activity first."""
        events = chat_store.load_events(self.chat_dir, day) if events is None else events
        clips = [c for c in (self.clip(e, day) for e in events) if c]
        prev = (Date.fromisoformat(day) - timedelta(days=1)).isoformat()
        if clips:  # an activity started before midnight: link to it, show only this day's part
            first = min(c.start for c in clips)
            reach = max(self.s.stitch_transition_gap_sec, self.s.stitch_track_gap_sec)
            clips += [c for c in (self.clip(e, prev) for e in chat_store.load_events(self.chat_dir, prev)) if c
                      and c.end >= first - reach]
        now = datetime.now(self.tz).timestamp()
        open_for = max(self.s.stitch_transition_gap_sec, self.s.stitch_continuation_gap_sec)
        groups = []
        for members, links in self.group(clips):
            mine = [c for c in members if c.day == day]
            if not mine:
                continue
            ids = {c.id for c in mine}
            start, end = mine[0].start, max(c.end for c in mine)
            groups.append({
                "id": mine[0].id,
                "events": [c.id for c in mine],
                "start": self._iso(start), "end": self._iso(end),
                "cameras": list(dict.fromkeys(c.camera for c in mine)),
                "links": [ln for ln in links if ln["a"] in ids and ln["b"] in ids],
                "clips": {c.id: {"camera": c.camera, "start_ms": int(c.start * 1000), "end_ms": int(c.end * 1000),
                                 "window": c.window} for c in mine},
                "open": now - end < open_for,
                "continued_from": prev if len(mine) < len(members) else None,
            })
        groups.sort(key=lambda g: (max(c["end_ms"] for c in g["clips"].values()), g["id"]), reverse=True)
        return groups

    def _iso(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, self.tz).isoformat(timespec="seconds")


# ---------- CLI ----------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--all", action="store_true", help="Also list single-clip activities")
    args = parser.parse_args(argv)
    setup_logging(PROGRAM, False, log.name)

    settings = load_settings()
    st = Stitcher(settings)
    day = args.date or datetime.now(st.tz).date().isoformat()
    groups = st.day(day)
    multi = [g for g in groups if len(g["events"]) > 1]
    log.info(f"Stitching {day}: {sum(len(g['events']) for g in groups)} event(s), {len(multi)} activities "
             f"with 2+ clips, {len(groups) - len(multi)} single clip(s)")
    for g in reversed(groups if args.all else multi):
        log.info(f"Activity {g['id']}: {g['start'][11:19]}-{g['end'][11:19]}, {len(g['events'])} clip(s), "
                 f"{' / '.join(g['cameras'])}{', continued from ' + g['continued_from'] if g['continued_from'] else ''}")
        for ln in g["links"]:
            log.info(f"  {ln['kind']:<12} {ln['a']} -> {ln['b']}: {ln['reason']}")
    return 0

