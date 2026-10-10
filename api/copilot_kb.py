"""
api/copilot_kb.py — Sekka's local knowledge: what an Egyptian co-driver knows that no trip context carries.

Retrieval, not a bigger prompt. Each entry is a few lines of durable, checkable fact about driving in Egypt
(emergency numbers, the rhythm of the week, Ramadan, fog and khamaseen, desert-road fuel, tolls, checkpoints,
car trouble). A turn gets at most MAX_ENTRIES entries, and only the ones its words (or the season / hour, for
the hazards that come with them) point at — so knowledge costs tokens only on the turns that need it, and the
static system prompt stays byte-identical across turns (the provider's prompt cache keeps it at a tenth of the
price).

Rules for an entry: no prices, no fines, no phone numbers other than the national emergency lines, nothing
that changes month to month. When Sekka does not know, it says so — the knowledge block says "use when
relevant", never "recite".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

MAX_ENTRIES = 2


@dataclass(frozen=True)
class Entry:
    key: str
    triggers: Tuple[str, ...]          # regex fragments, matched on the normalized utterance
    fact: str


def _n(s: str) -> str:
    s = (s or "").lower()
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي"), ("ـ", "")):
        s = s.replace(a, b)
    return " ".join(s.split())


ENTRIES: List[Entry] = [
    Entry("emergency", (r"emergenc", r"ambulanc", r"police", r"fire\b", r"injur", r"bleed", r"crash",
                        r"اسعاف", r"نجده", r"مطافي", r"مطافئ", r"طوارئ", r"متعور", r"اتعور", r"اتخبط",
                        r"حريقه", r"بوليس", r"شرطه"),
          "Egypt's national emergency lines: police (النجدة) 122, ambulance (الإسعاف) 123, fire (المطافي) 180. "
          "The emergency_call tool dials one after a single yes. In a crash: hazards on, get people away from "
          "traffic, call 123 for injuries."),
    Entry("rush", (r"\brush", r"best time", r"when (should|to) (i )?(leave|go)", r"busy", r"peak",
                   r"وقت الذروه", r"امتي (الطريق|الشوارع)", r"امشي امتي", r"اتحرك امتي", r"زحمه النهارده",
                   r"الشوارع فاضيه", r"الطريق فاضي"),
          "Cairo's weekday rhythm (Sunday–Thursday): heavy roughly 7:30–10:30 and 14:00–19:00; Thursday evening "
          "is the worst of the week; Friday morning until the noon prayer is the emptiest; Saturday is lighter "
          "than a workday. The trip data's live traffic always wins over this rhythm."),
    Entry("ramadan", (r"ramadan", r"iftar", r"suhoor", r"رمضان", r"فطار", r"الفطار", r"سحور", r"المغرب",
                      r"مدفع"),
          "In Ramadan the heaviest traffic is the last one to two hours before iftar (Maghrib); the roads go "
          "almost empty right at iftar and fill again later at night."),
    Entry("coast", (r"sahel", r"north coast", r"alex", r"sokhna", r"ساحل", r"اسكندري", r"السخنه",
                    r"العين السخنه", r"مرسي مطروح", r"العلمين", r"الجونه", r"الغردقه"),
          "Summer weekends load the coast roads: the Cairo–Alexandria desert road, the North Coast road and the "
          "Ain Sokhna road run heavy outbound Thursday afternoon/Friday morning and inbound Saturday evening."),
    Entry("fog", (r"\bfog", r"visib", r"شبوره", r"ضباب"),
          "Winter mornings (December–February) bring dense fog (الشبورة) on desert and agricultural roads, "
          "worst around dawn: slow down, low beams and fog lights, long gaps; highways are sometimes closed for "
          "a few hours for fog."),
    Entry("khamaseen", (r"\bsand", r"dust", r"storm", r"\bwindy?\b", r"خماسين", r"رمله", r"تراب", r"عاصفه", r"هوا جامد"),
          "Spring (March–May) khamaseen dust storms can cut visibility on desert roads to a few tens of metres: "
          "headlights on, slow down, never stop in a lane."),
    Entry("fuel", (r"\bfuel", r"petrol", r"gas(oline)?\b", r"octane", r"diesel", r"benzin", r"بنزين", r"بنزينه",
                   r"سولار", r"تفويل", r"اموّن", r"\bامون"),
          "Egyptian grades: gasoline 80, 92 and 95 octane (بنزين ٨٠ / ٩٢ / ٩٥), diesel is «سولار»; most modern "
          "cars take 92 or 95. On long desert roads stations can be 50 km or more apart — fill up before leaving "
          "town. Prices are set by the state and change: never quote one."),
    Entry("tolls", (r"\btolls?\b", r"\bgates?\b", r"كارته", r"كارتة", r"بوابه", r"بوابات", r"رسوم"),
          "Toll gates (الكارتة / البوابات) stand on the Cairo–Alexandria desert road, the Regional Ring Road, "
          "the Cairo–Ain Sokhna road and several new-city roads. Amounts change — never quote one; "
          "route_options can avoid tolls."),
    Entry("papers", (r"licen[cs]e", r"checkpoint", r"papers", r"رخصه", r"رخص", r"كمين", r"لجنه", r"ورق العربيه"),
          "Carry both the driving licence (رخصة القيادة) and the car licence (رخصة العربية): highway "
          "checkpoints (الكمين) ask for both. Seat belts are enforced; holding the phone while driving is an "
          "offence."),
    Entry("bumps", (r"bump", r"village", r"microbus", r"مطب", r"مطبات", r"قريه", r"ميكروباص", r"توكتوك", r"تكاتك"),
          "Unmarked speed bumps (المطبات) are common entering towns and villages and on regional roads; "
          "microbuses stop suddenly for passengers and tuk-tuks join from side streets — ease off near them."),
    Entry("night", (r"night", r"\bdark", r"high beam", r"بالليل", r"\bليل", r"ضلمه", r"النور العالي", r"الكشافات"),
          "At night unlit agricultural and regional roads hide slow trucks, tractors and carts without lights, "
          "and oncoming cars often keep high beams on: look to the right edge of your lane, slow down."),
    Entry("parking", (r"\bpark", r"sayes", r"\btow(ed|ing)?\b", r"ركن", r"اركن", r"سايس", r"ونش", r"جراج"),
          "Street parking in Cairo is often run by an informal attendant (السايس) who expects a small tip; "
          "blocking a gate or a no-parking kerb risks the tow truck (الونش). save_parking remembers the spot."),
    Entry("breakdown", (r"flat", r"tyre", r"tire", r"puncture", r"overheat", r"broke down", r"breakdown",
                        r"battery", r"كاوتش", r"فرقع", r"بنشر", r"سخنت", r"حراره", r"عطلت", r"اتعطلت",
                        r"بطاريه", r"ونش"),
          "Flat tyre: a «بنشر» (tyre shop) is usually close in any town — find_places can find one. On a highway "
          "stop fully on the shoulder, hazards on, warning triangle well behind the car. Overheating: pull over, "
          "engine off, never open a hot radiator cap."),
    Entry("airport", (r"airport", r"terminal", r"flight", r"مطار", r"طياره"),
          "Cairo International Airport has terminals 1, 2 and 3 (2 and 3 sit side by side); departures kerbs are "
          "drop-off only. Leave early at peak hours — the airport road and Salah Salem jam."),
    Entry("friday", (r"friday", r"prayer", r"mosque", r"الجمعه", r"صلاه", r"\bجامع\b", r"مسجد"),
          "Around Friday noon prayer (about 12:00–13:00) streets beside big mosques are blocked by worshippers' "
          "cars; the rest of Friday morning is the quietest traffic of the week."),
    Entry("heat", (r"\bheat", r"\bhot\b", r"summer", r"\bac\b", r"\bحر\b", r"الصيف", r"التكييف"),
          "Summer afternoons pass 35–40 °C in Cairo and higher in Upper Egypt: tyre pressure climbs, engines run "
          "hot in jams, and a parked car becomes dangerous for children and pets within minutes."),
    Entry("fatigue", (r"tired", r"sleepy", r"drowsy", r"exhausted", r"\bnap\b", r"نعسان", r"نعسانه", r"تعبان",
                      r"تعبانه", r"هموت من النوم", r"عيني بتقفل", r"مرهق", r"راحه", r"استراحه"),
          "Drowsy driving: a 15–20 minute break (coffee, a short walk) beats pushing on; on long roads stop about "
          "every two hours. Stop at a station, a rest house or a cafe — never on a highway shoulder except in an "
          "emergency. find_places along_route can find the next one."),
    Entry("rain", (r"\brain", r"flood", r"underpass", r"tunnel", r"مطر", r"مطره", r"شتا", r"غرق", r"ميه في",
                   r"نفق", r"انفاق", r"نوه", r"نوات"),
          "Rain is rare but heavy when it comes: underpasses and tunnels (الأنفاق) and low streets in Cairo and "
          "Alexandria flood fast — never drive into standing water of unknown depth. Alexandria's winter storms "
          "(النوات) run November–March; the first rain makes roads slick with oil."),
    Entry("minor_crash", (r"scratch", r"fender", r"bumped", r"hit (my|the) car", r"insurance", r"خبطه", r"خبطني",
                          r"خبطت", r"تأمين", r"تامين", r"محضر", r"قسم"),
          "A minor collision with nobody hurt: move out of the lane if the cars can move, hazards on, photograph "
          "both cars and plates, exchange licence details; insurance claims need a police report (محضر) at the "
          "nearest police station (القسم). Anyone hurt → 123 first."),
    Entry("fines", (r"\bfine", r"ticket", r"violation", r"مخالف", r"غرامه", r"غرامات", r"سحب الرخصه"),
          "Traffic violations (radar, parking, papers) are attached to the car licence and are checked and paid "
          "online through the government's digital services (مصر الرقمية) or at the traffic unit (وحدة المرور) "
          "before renewal. Amounts change — never quote one."),
    Entry("uturn", (r"u.?turn", r"turn around", r"go back", r"service road", r"ملف", r"يلف", r"ارجع ورا",
                    r"طريق الخدمه", r"الخدمه", r"دوران"),
          "On Egyptian arterials many left turns are made through a U-turn opening (الملف) past the junction, so "
          "a route can pass the destination and come back. Highways often run a parallel service road (طريق "
          "الخدمة): miss its entry and the next U-turn fixes it."),
    Entry("restroom", (r"restroom", r"bathroom", r"toilet", r"\bwc\b", r"حمام", r"تواليت", r"دوره ميه"),
          "Clean restrooms on the road: the big fuel stations with a shop and the rest houses on intercity roads, "
          "and in the city the malls. find_places along_route («بنزينة» or «مول») finds one."),
    Entry("motorbikes", (r"motorbike", r"motorcycle", r"delivery", r"scooter", r"موتوسيكل", r"موتوسيكلات",
                         r"دليفري", r"اسكوتر"),
          "Delivery motorbikes weave between lanes and ride against the traffic on side streets — check the "
          "mirrors and the blind spot before every lane change and turn."),
    Entry("pedestrians", (r"pedestrian", r"crossing", r"walker", r"مشاه", r"ماشيين", r"بيعدي", r"بيعدوا",
                          r"عابرين"),
          "Pedestrians cross wherever they need to — including the Ring Road and highways, most near microbus "
          "stops, bridges and markets, and hardest to see at dusk. Ease off near crowds."),
    Entry("school", (r"school", r"مدرسه", r"مدارس", r"الدراسه", r"اتوبيس المدرسه"),
          "On school days the streets around schools jam roughly 7:00–8:00 and 13:30–15:00, with buses double-"
          "parked; the school year runs about late September to June."),
    Entry("ev", (r"\bev\b", r"electric car", r"charg", r"tesla", r"عربيه كهربا", r"شاحن", r"شحن"),
          "EV chargers in Egypt are concentrated in Cairo's malls, some fuel stations and the new cities; long "
          "intercity runs need planning. find_places can search «شاحن عربيات كهربا» / 'EV charger'."),
    Entry("speed", (r"speed limit", r"how fast", r"\bradar", r"السرعه", r"السرعة", r"رادار", r"اقصي سرعه"),
          "Speed radars stand on the Ring Road, the highways and many city arterials, often right after a curve "
          "or a bridge; the posted limit (and the trip data's) is the one that counts. Mobile radars move."),
]

_COMPILED = [(e, re.compile("|".join(e.triggers))) for e in ENTRIES]


def lookup(text: str, ctx: Optional[Dict[str, Any]] = None) -> List[Entry]:
    """The entries this utterance points at (≤ MAX_ENTRIES), best first."""
    t = _n(text)
    if not t:
        return []
    scored = []
    for e, rx in _COMPILED:
        hits = len(rx.findall(t))
        if hits:
            scored.append((hits, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:MAX_ENTRIES]]


def block(text: str, ctx: Optional[Dict[str, Any]] = None) -> str:
    """The prompt block for this turn, or "" (no knowledge needed — no tokens spent)."""
    got = lookup(text, ctx)
    if not got:
        return ""
    return ("\n[Local knowledge — use only if it answers the driver; the trip data wins on any conflict]\n"
            + "\n".join(f"- {e.fact}" for e in got))
