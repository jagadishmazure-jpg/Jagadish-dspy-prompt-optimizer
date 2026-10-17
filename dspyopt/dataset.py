"""Small synthetic tasks with train/validation splits.

All data is fictional and generated for this repo. Each task returns
{"train": [...], "val": [...]} where an example is
{"input": str, "output": str}. Train and validation inputs are disjoint.

Sentiment labels: "positive" / "negative".
Extract outputs: JSON strings like '{"priority": "high", "category": "billing"}'.
"""

from __future__ import annotations

import json
from typing import Dict, List


def build_oracle(examples: List[Dict[str, str]]) -> Dict[str, str]:
    """Map input text -> gold output (feeds FakeLM)."""
    return {ex["input"]: ex["output"] for ex in examples}


def _ticket(text: str, priority: str, category: str) -> Dict[str, str]:
    return {
        "input": text,
        "output": json.dumps({"priority": priority, "category": category}, sort_keys=True),
    }


def sentiment_task() -> Dict[str, List[Dict[str, str]]]:
    train = [
        {"input": "I absolutely love this blender, it works perfectly every morning.", "output": "positive"},
        {"input": "Fantastic phone, battery lasts two days and the camera is superb.", "output": "positive"},
        {"input": "Great value for the price, highly recommend to anyone.", "output": "positive"},
        {"input": "The shoes are comfortable and arrived earlier than expected.", "output": "positive"},
        {"input": "Terrible headphones, the sound cuts out constantly.", "output": "negative"},
        {"input": "Worst purchase ever, broke within a week of normal use.", "output": "negative"},
        {"input": "Awful customer service and the product never worked.", "output": "negative"},
        # hard: mixed signals — the overall judgment decides
        {"input": "The design is beautiful but the battery life is disappointing.", "output": "negative"},
        {"input": "I liked the idea, however the execution was poor and buggy.", "output": "negative"},
        {"input": "Great packaging and fast shipping, although the item itself feels cheap.", "output": "negative"},
        # tricky: sarcasm
        {"input": "Oh great, another update that makes everything slower. Just what I wanted.", "output": "negative"},
        {"input": "Yeah right, like this 'premium' cable was worth the price.", "output": "negative"},
    ]
    val = [
        {"input": "Love the fit and the fabric feels premium.", "output": "positive"},
        {"input": "Excellent laptop, fast and reliable for my work.", "output": "positive"},
        {"input": "Disappointing quality, would not buy again.", "output": "negative"},
        {"input": "The app crashes daily and support never replies.", "output": "negative"},
        {"input": "Nice look and feel, but it stopped working after a month.", "output": "negative"},
        {"input": "The staff were friendly, however the food was cold and bland.", "output": "negative"},
        {"input": "Sarcastic applause for the slowest delivery I have ever seen.", "output": "negative"},
        {"input": "Solid build quality, does exactly what it promises.", "output": "positive"},
    ]
    return {"train": train, "val": val}


def extract_task() -> Dict[str, List[Dict[str, str]]]:
    train = [
        _ticket("My order #1234 never arrived, but tracking shows it was delivered to the wrong address.", "high", "shipping"),
        _ticket("How do I reset my password for the dashboard?", "low", "account"),
        _ticket("The checkout page throws an error every time I try to pay.", "high", "billing"),
        _ticket("Can I change the delivery address on order #5678?", "low", "shipping"),
        _ticket("Not sure if this is urgent, but I think I was double charged on my last invoice.", "high", "billing"),
        _ticket("Maybe the app is just slow today, it takes a minute to load my reports.", "low", "performance"),
        _ticket("Where can I download my receipts for last quarter?", "low", "account"),
        _ticket("Our team might be locked out after the migration, nobody can sign in.", "high", "account"),
    ]
    val = [
        _ticket("Package arrived damaged and several items are missing.", "high", "shipping"),
        _ticket("Please update my email address on file.", "low", "account"),
        _ticket("The refund I was promised two weeks ago still has not arrived.", "high", "billing"),
        _ticket("Is there a dark mode for the mobile app?", "low", "account"),
        _ticket("Not sure this matters, but my card was charged twice for the same subscription.", "high", "billing"),
        _ticket("The export feature might be broken, my CSV came out empty.", "low", "performance"),
    ]
    return {"train": train, "val": val}
