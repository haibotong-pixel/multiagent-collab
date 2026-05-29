from __future__ import annotations

from dataclasses import dataclass

from macollab.tasks.base import Task

_PROBLEMS: list[tuple[str, int]] = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. "
     "How many dollars did she earn?", 10),
    ("Betty is saving for a $100 wallet. She has half of the money she needs. Her "
     "parents give her $15 and her grandparents twice as much as her parents. How "
     "much more money does Betty need to buy the wallet?", 5),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
     "bolts in total does it take?", 3),
    ("James writes a 3-page letter to 2 different friends twice a week. How many "
     "pages does he write a year?", 624),
    ("Mark has a garden with flowers. He planted 10 yellow, 80% more purple than "
     "yellow, and 25% as many green as the combined yellow and purple. How many "
     "flowers does Mark have in his garden?", 35),
    ("A store had 120 apples. They sold 45 in the morning and 30 in the afternoon. "
     "How many apples are left?", 45),
    ("Tom buys 4 books that cost $7 each and a pen that costs $3. How much does he "
     "spend in total?", 31),
    ("A train travels 60 miles per hour for 3 hours, then 40 miles per hour for 2 "
     "hours. How many miles does it travel in total?", 260),
    ("Sara has 5 boxes with 12 pencils each. She gives away 18 pencils. How many "
     "pencils does she have left?", 42),
]


@dataclass
class Gsm8kMiniSuite:
    name: str = "gsm8k_mini"

    def tasks(self) -> list[Task]:
        return [
            Task(id=f"gsm8k_mini_{i:02d}", prompt=p, ground_truth=ans, type="reasoning")
            for i, (p, ans) in enumerate(_PROBLEMS)
        ]
