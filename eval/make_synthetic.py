"""Generate a medium-sized synthetic labeled dataset for run_eval.py.

Cases are built from templates whose labels are unambiguous by construction, across four
domains: support routing, review sentiment, content moderation and agent next-step.

    uv run python eval/make_synthetic.py --n 150 --out eval/datasets/synthetic.jsonl

The same --seed always produces the same file.
"""

import argparse
import json
import random
from pathlib import Path

DEPARTMENTS = {
    "billing": "Payments, invoices, refunds, pricing",
    "technical": "Bugs, errors, outages, integrations",
    "account": "Login, password, profile, account access or deletion",
    "sales": "New purchases, upgrades, quotes, demos",
}
URGENCY = [
    "Low: no time pressure",
    "Medium: should be handled today",
    "High: business is blocked or money is being lost right now",
]

# (text, refund, department, urgency)
SUPPORT = [
    (
        "I was charged twice for my {month} invoice. Please refund the extra {amount}.",
        True,
        "billing",
        1,
    ),
    (
        "Can I get my money back? I cancelled in {month} but you still billed me {amount}.",
        True,
        "billing",
        1,
    ),
    (
        "Could you send me a copy of the {month} invoice for our accountant? No rush.",
        False,
        "billing",
        0,
    ),
    (
        "What is the price difference between the Basic and Pro plans? Just curious.",
        False,
        "sales",
        0,
    ),
    (
        "We'd like a quote for {seats} seats on the Enterprise plan, and a demo next week "
        "if possible.",
        False,
        "sales",
        0,
    ),
    (
        "Your API has returned 500 errors for the last {hours} hours and our checkout is down. "
        "We are losing orders right now!",
        False,
        "technical",
        2,
    ),
    (
        "The {feature} page shows a blank screen in Firefox. "
        "Chrome works fine, so it's not urgent.",
        False,
        "technical",
        0,
    ),
    (
        "Our {feature} integration stopped syncing this morning. We need it working before "
        "end of day.",
        False,
        "technical",
        1,
    ),
    (
        "I forgot my password and the reset email never arrives. I can't log in to finish "
        "today's report.",
        False,
        "account",
        1,
    ),
    ("Please delete my account and all my personal data.", False, "account", 0),
    (
        "Someone changed the email on my account without my permission and is logged in right now! "
        "Lock it immediately.",
        False,
        "account",
        2,
    ),
    (
        "Production is completely down for all {seats} of our users. Nothing loads.",
        False,
        "technical",
        2,
    ),
    (
        "The {feature} export is broken and corrupted our data. I want a full refund for "
        "this month.",
        True,
        "technical",
        1,
    ),
]

# (text, positive, stars 0..4)
REVIEWS = [
    (
        "Absolutely love this {product}. Works perfectly, best purchase I've made this year.",
        True,
        4,
    ),
    ("Great {product}, does what it says. Shipping was a bit slow but I'm happy.", True, 3),
    ("It's okay. The {product} works but feels cheap. Nothing special either way.", False, 2),
    (
        "Disappointed. The {product} stopped working after {days} days and support ignored me.",
        False,
        1,
    ),
    (
        "Total garbage. The {product} arrived broken, smelled burnt "
        "and the seller refused a refund. Avoid.",
        False,
        0,
    ),
    ("Five stars! My whole family uses this {product} every day now.", True, 4),
    ("Returned it. The {product} was nothing like the photos.", False, 1),
]

# (text, violates, category)
MODERATION_CATS = {
    "none": "No policy violation",
    "harassment": "Insults or attacks aimed at a person",
    "spam": "Advertising, scams, repeated promotional links",
    "violence": "Threats of physical harm",
}
MODERATION = [
    ("Does anyone know a good recipe for {food}? Thanks in advance!", False, "none"),
    ("Great point in the article, I hadn't thought about {topic} that way.", False, "none"),
    ("You are a pathetic idiot and everyone here hates you, {name}.", True, "harassment"),
    ("BUY CHEAP {product} NOW!!! 90% OFF click www.deals-{n}.biz www.deals-{n}.biz", True, "spam"),
    ("Earn $5000 a week from home, DM me for the secret crypto method!!!", True, "spam"),
    (
        "I know where you live, {name}. Say that again and I'll come break your legs.",
        True,
        "violence",
    ),
    ("I disagree with your take on {topic}, but I appreciate the sources.", False, "none"),
]

AGENT_STEPS = {
    "answer": "Reply to the user directly with the information already available",
    "search": "Search the knowledge base or web for missing information",
    "ask": "Ask the user a clarifying question",
    "escalate": "Hand off to a human agent",
}
# (text, needs_tool, step)
AGENT = [
    ("User: What is 2 + 2?", False, "answer"),
    ("User: Thanks, that solved it!", False, "answer"),
    ("User: What was the closing price of {ticker} stock yesterday?", True, "search"),
    ("User: What's the latest news about {topic}?", True, "search"),
    ("User: Can you book it?", False, "ask"),
    ("User: Fix the thing from before.", False, "ask"),
    ("User: I've asked three times, I want to speak to a real person NOW.", False, "escalate"),
    (
        "User: I'm a lawyer representing a client suing your company; who do I contact?",
        False,
        "escalate",
    ),
]

FILL = {
    "month": ["January", "March", "June", "September", "November"],
    "amount": ["$19", "$49.99", "€120", "$300"],
    "seats": ["25", "120", "500", "2,000"],
    "hours": ["two", "three", "six"],
    "feature": ["reports", "billing dashboard", "Salesforce", "Slack", "CSV"],
    "product": ["blender", "headphones", "backpack", "desk lamp", "coffee maker"],
    "days": ["three", "ten", "twenty"],
    "food": ["lasagna", "banana bread", "pho", "falafel"],
    "topic": ["remote work", "electric cars", "interest rates", "city planning"],
    "name": ["Sam", "Alex", "Jordan", "Riley"],
    "n": ["7", "42", "88"],
    "ticker": ["AAPL", "NVDA", "MSFT"],
}


def fill(rng: random.Random, text: str) -> str:
    return text.format(**{k: rng.choice(v) for k, v in FILL.items()})


def support(rng, i):
    text, refund, dept, urg = rng.choice(SUPPORT)
    return {
        "id": f"support-{i:03d}",
        "state": fill(rng, text),
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for a refund or their money back?",
            },
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": DEPARTMENTS,
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this message?",
                "criteria": URGENCY,
            },
        },
        "labels": {"refund": refund, "department": dept, "urgency": urg},
    }


def review(rng, i):
    text, positive, stars = rng.choice(REVIEWS)
    return {
        "id": f"review-{i:03d}",
        "state": fill(rng, text),
        "questions": {
            "positive": {"type": "noul", "instructions": "Is the review overall positive?"},
            "stars": {
                "type": "score",
                "instructions": "What star rating fits this review?",
                "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
            },
        },
        "labels": {"positive": positive, "stars": stars},
    }


def moderation(rng, i):
    text, violates, cat = rng.choice(MODERATION)
    return {
        "id": f"moderation-{i:03d}",
        "state": fill(rng, text),
        "questions": {
            "violates": {
                "type": "noul",
                "instructions": "Does this post violate community guidelines?",
            },
            "category": {
                "type": "choice",
                "instructions": "Which category best fits the post?",
                "criteria": MODERATION_CATS,
            },
        },
        "labels": {"violates": violates, "category": cat},
    }


def agent(rng, i):
    text, needs_tool, step = rng.choice(AGENT)
    return {
        "id": f"agent-{i:03d}",
        "state": fill(rng, text),
        "questions": {
            "needs_tool": {
                "type": "noul",
                "instructions": "Does answering require looking up external or "
                "real-time information?",
            },
            "next_step": {
                "type": "choice",
                "instructions": "What should the assistant do next?",
                "criteria": AGENT_STEPS,
            },
        },
        "labels": {"needs_tool": needs_tool, "next_step": step},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="eval/datasets/synthetic.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    makers = [support, review, moderation, agent]
    cases = [makers[i % len(makers)](rng, i) for i in range(args.n)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c) + "\n" for c in cases))
    n_questions = sum(len(c["questions"]) for c in cases)
    print(f"wrote {len(cases)} cases ({n_questions} questions) to {out}")


if __name__ == "__main__":
    main()
