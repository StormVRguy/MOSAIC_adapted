#!/usr/bin/env python3
"""Generate trial_reflections.csv for MOSAIC pipeline (readme guidelines)."""
import csv
import random
from pathlib import Path

# Reflections: max 70 characters ignoring spaces (readme)
def char_count(s):
    return len(s.replace(" ", ""))

AFFECTION = [
    "I felt so much love today.",
    "Hugging my kids made everything better.",
    "Gratitude for my partner's support.",
    "Family dinner brought us closer.",
    "A smile from a stranger warmed my heart.",
    "Love and kindness go a long way.",
    "I told them how much I care.",
    "We celebrated our anniversary quietly.",
    "My dog greeted me with pure joy.",
    "Friends checked in when I was low.",
    "I felt grateful for my family.",
    "A small gesture meant a lot.",
    "We laughed together until we cried.",
    "Holding hands felt comforting.",
    "I missed my parents today.",
    "The card from my sister was sweet.",
    "I tried to show more patience.",
    "We had a cozy evening at home.",
    "I felt loved and supported.",
    "Kind words from a colleague helped.",
    "My niece's drawing made me smile.",
    "I called my best friend.",
    "We shared a meal and stories.",
    "I felt connected to others today.",
    "Their encouragement kept me going.",
    "I wrote a thank-you note.",
    "We hugged and made up.",
    "I appreciated the small moments.",
    "Love makes everything feel lighter.",
    "I felt thankful for my health.",
    "We spent quality time together.",
    "A compliment made my day.",
    "I tried to be more present.",
    "Family game night was fun.",
    "I felt the warmth of community.",
    "Saying I love you felt right.",
    "We supported each other through it.",
]

ENGINEERING = [
    "The new design passed the stress test.",
    "Debugging took hours but we fixed it.",
    "The prototype works better than expected.",
    "Code review caught a critical bug.",
    "The bridge design is elegant and strong.",
    "We shipped the feature on time.",
    "The algorithm optimization paid off.",
    "The system handled the load well.",
    "We refactored the legacy module.",
    "The test suite gives me confidence.",
    "The circuit layout is much cleaner now.",
    "Documentation saved us hours.",
    "The build finally passed.",
    "We reduced latency by forty percent.",
    "The new framework is a good fit.",
    "The deployment went smoothly.",
    "We fixed the memory leak.",
    "The API design is consistent now.",
    "The sensor calibration worked.",
    "We solved the integration issue.",
    "The database query is optimized.",
    "The mechanical part fits perfectly.",
    "We improved the error handling.",
    "The simulation matches the lab results.",
    "The pipeline runs faster now.",
    "We documented the edge cases.",
    "The thermal design is stable.",
    "The code is more maintainable.",
    "We met the performance target.",
    "The prototype passed user testing.",
    "The architecture scales well.",
    "We resolved the dependency conflict.",
    "The weld passed inspection.",
    "The new tool saved us time.",
    "The design review was positive.",
    "The server stayed up under load.",
    "We automated the boring parts.",
]

SPORTS = [
    "We won the match in the final minutes.",
    "Training was tough but rewarding.",
    "The goal was incredible.",
    "Team spirit carried us through.",
    "I beat my personal best today.",
    "The coach gave great feedback.",
    "We practiced set pieces for hours.",
    "The crowd cheered at the end.",
    "I felt strong in the last mile.",
    "We came back from two goals down.",
    "The new shoes helped my time.",
    "Team breakfast boosted morale.",
    "I finally nailed the technique.",
    "We supported each other throughout.",
    "The match was intense and fair.",
    "I recovered well after the race.",
    "We celebrated with the fans.",
    "The training plan is working.",
    "I stayed focused under pressure.",
    "We worked on defense all week.",
    "The referee was consistent.",
    "I felt the adrenaline kick in.",
    "We qualified for the next round.",
    "The injury is healing well.",
    "I ran my fastest lap yet.",
    "We analyzed the opponent's tactics.",
    "The stadium atmosphere was electric.",
    "I trained with the squad today.",
    "We shared the victory together.",
    "The warm-up routine paid off.",
    "I improved my breathing technique.",
    "We kept our discipline until the end.",
    "The new play worked perfectly.",
    "I felt part of something bigger.",
    "We gave everything we had.",
    "The final sprint was exhausting.",
    "Team talk before the game helped.",
]

ALL = [(r, "affection") for r in AFFECTION] + [(r, "engineering") for r in ENGINEERING] + [(r, "sports") for r in SPORTS]
MAX_CHARS = 70

# Enforce max length (ignoring spaces)
def trim(s):
    s = s.strip()
    if char_count(s) <= MAX_CHARS:
        return s
    out = []
    n = 0
    for c in s:
        if c != " ":
            n += 1
            if n > MAX_CHARS:
                break
        out.append(c)
    return "".join(out).strip()

trimmed = [(trim(r), label) for r, label in ALL]
random.seed(42)
random.shuffle(trimmed)

out_dir = Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / "trial_cleaned.csv"

with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["reflection_answer", "topic"])  # topic is optional, pipeline uses reflection_answer
    for text, topic in trimmed:
        w.writerow([text, topic])

print(f"Wrote {len(trimmed)} rows to {path}")
print(f"Column: reflection_answer (topic kept for reference).")
assert all(char_count(r[0]) <= MAX_CHARS for r in trimmed), "Max length violated"
