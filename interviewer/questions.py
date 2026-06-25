from __future__ import annotations

"""
Behavioral question bank, organised by category.
The session picks questions either from a JD-driven LLM selection or
randomly across categories to cover diverse competencies.
"""

import random

QUESTIONS: dict[str, list[str]] = {
    "self_intro": [
        "Tell me about yourself and why you're interested in this role.",
        "Walk me through your background and what brought you here today.",
        "Give me a two-minute overview of your career so far.",
    ],
    "leadership": [
        "Tell me about a time you led a project under a tight deadline.",
        "Describe a situation where you had to influence a team without formal authority.",
        "Tell me about a time you had to make a tough decision with incomplete information.",
        "Describe a moment when you had to step up and take ownership of a failing project.",
        "Tell me about a time you mentored or coached someone on your team.",
    ],
    "conflict": [
        "Tell me about a time you disagreed with a manager or senior colleague. How did you handle it?",
        "Describe a situation where you had a conflict with a teammate. How was it resolved?",
        "Tell me about a time you had to deliver difficult feedback to someone.",
        "Describe a time when you had competing priorities and how you managed stakeholder expectations.",
    ],
    "problem_solving": [
        "Tell me about the most technically challenging problem you've solved.",
        "Describe a time when you had to learn something completely new to solve a problem.",
        "Tell me about a time you identified a bug or inefficiency that no one else had noticed.",
        "Walk me through a situation where your initial approach failed and how you pivoted.",
        "Tell me about a time you had to debug a complex production issue under pressure.",
    ],
    "collaboration": [
        "Tell me about a time you worked effectively in a cross-functional team.",
        "Describe a successful project you delivered as part of a team. What was your specific contribution?",
        "Tell me about a time you had to rely on others outside your team to get something done.",
        "Describe how you've handled working with teammates who have very different working styles.",
    ],
    "failure": [
        "Tell me about a project or decision that didn't go as planned. What did you learn?",
        "Describe a time you made a mistake that had a real impact. How did you handle it?",
        "Tell me about a goal you set for yourself that you didn't achieve. What happened?",
        "Walk me through a time you shipped something that caused problems. What did you do next?",
    ],
    "achievement": [
        "What's the professional achievement you're most proud of in the last two years?",
        "Tell me about a time you went significantly above and beyond what was expected of you.",
        "Describe a time you improved a process or system that had a measurable impact.",
        "Tell me about a time you delivered results that surprised your team or manager.",
    ],
    "adaptability": [
        "Tell me about a time you had to adapt to a major change at work.",
        "Describe a situation where the requirements changed mid-project. How did you handle it?",
        "Tell me about a time you had to work in an area completely outside your comfort zone.",
        "Describe a time when your priorities shifted dramatically. How did you reprioritise?",
    ],
    "motivation": [
        "Why are you looking to make a change right now?",
        "What kind of work environment brings out the best in you?",
        "Tell me about a project or task that you found genuinely exciting. What made it engaging?",
        "Where do you see yourself in three years, and how does this role fit into that?",
    ],
    "communication": [
        "Tell me about a time you had to explain a complex technical concept to a non-technical stakeholder.",
        "Describe a situation where miscommunication caused a problem. What would you do differently?",
        "Tell me about a time you had to present your work to senior leadership.",
        "Describe a time when you had to write a document or proposal that changed someone's mind.",
    ],
}

ALL_QUESTIONS: list[tuple[str, str]] = [
    (category, q)
    for category, qs in QUESTIONS.items()
    for q in qs
]


def pick_questions(n: int = 10, seed: int | None = None) -> list[str]:
    """
    Pick n questions that cover diverse categories.
    Always starts with a self_intro question, then samples from the rest.
    """
    rng = random.Random(seed)

    # Always open with self-intro
    intro = rng.choice(QUESTIONS["self_intro"])
    remaining_categories = [c for c in QUESTIONS if c != "self_intro"]

    # Round-robin across categories to ensure coverage
    selected: list[str] = [intro]
    category_pool = remaining_categories * 3  # ensure enough to sample from
    rng.shuffle(category_pool)
    seen_categories: set[str] = {"self_intro"}

    for cat in category_pool:
        if len(selected) >= n:
            break
        qs = QUESTIONS[cat]
        q = rng.choice(qs)
        if q not in selected:
            selected.append(q)
            seen_categories.add(cat)

    return selected[:n]


def pick_questions_from_jd(jd: str, n: int = 10) -> list[str]:
    """
    Return a prompt fragment for the LLM to select relevant questions from the bank.
    The actual selection happens in the session via an LLM call.
    This just formats the bank for the prompt.
    """
    lines = ["Here is the full question bank. Select the most relevant questions for the role described.\n"]
    for i, (cat, q) in enumerate(ALL_QUESTIONS):
        lines.append(f"{i+1}. [{cat}] {q}")
    return "\n".join(lines)
