"""
generate_data.py

This script creates fake customer survey responses and saves them as a JSON
file, following the same shape as Appendix A in the assignment.

Why not just use totally random text?
Because then DataAgent would have nothing real to find. Instead we pick a
"theme" (wait time, food quality, staff, price, cleanliness) for each fake
response and write a review sentence that matches both the rating and the
theme. This way things like "top complaint themes" or "CSAT dropped last
month" actually show up in the numbers, instead of being pure noise.

Run it like this:
    python generate_data.py --count 8000
"""

import argparse
import json
import random
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

BUSINESSES = [
    {"business_id": "b01", "business_name": "GreenLeaf Bistro"},
    {"business_id": "b02", "business_name": "QuickFit Gym"},
    {"business_id": "b03", "business_name": "Urban Cuts Barbershop"},
]

SURVEYS = [
    {"survey_id": "s01", "survey_name": "Order Experience"},
    {"survey_id": "s02", "survey_name": "Membership Value"},
    {"survey_id": "s03", "survey_name": "Overall Satisfaction"},
]

CHANNELS = ["mobile", "web", "in_store_kiosk", "email"]

# One theme = one thing customers commonly talk about.
# Each theme has a few sentence templates for good ratings and a few for bad
# ratings, so the free_text roughly matches the numeric rating.
THEME_TEMPLATES = {
    "wait_time": {
        "positive": [
            "Service was quick, barely any wait at all.",
            "Got my order fast, no complaints on wait time.",
            "In and out quickly, really appreciated that.",
        ],
        "negative": [
            "The wait time was way too long today.",
            "Waited almost 25 minutes just to get served.",
            "Line moved so slowly, wait time needs work.",
        ],
    },
    "food_quality": {
        "positive": [
            "The food was great and tasted fresh.",
            "Really enjoyed the meal, quality was excellent.",
            "Food quality has been consistently good lately.",
        ],
        "negative": [
            "The food was cold and quality felt off.",
            "Quality has gone down, my order was undercooked.",
            "Not happy with the food quality this visit.",
        ],
    },
    "staff": {
        "positive": [
            "Staff was friendly and very helpful.",
            "The team member who helped me was great.",
            "Staff went out of their way to help, loved it.",
        ],
        "negative": [
            "Staff seemed rushed and not very friendly.",
            "One of the staff members was rude to me.",
            "Felt ignored by staff during my visit.",
        ],
    },
    "price": {
        "positive": [
            "Prices are fair for what you get.",
            "Good value for money, will come back.",
            "Reasonably priced compared to other places.",
        ],
        "negative": [
            "Prices feel too high for the portion size.",
            "Not worth the price honestly.",
            "Price went up but quality did not improve.",
        ],
    },
    "cleanliness": {
        "positive": [
            "Place was clean and well maintained.",
            "Everything looked spotless during my visit.",
            "Cleanliness has been great every time I visit.",
        ],
        "negative": [
            "Tables were dirty and not cleaned up.",
            "Cleanliness was a real issue this time.",
            "Restrooms were not clean at all.",
        ],
    },
}

THEMES = list(THEME_TEMPLATES.keys())


def pick_rating_and_sentiment():
    """
    Pick a star rating (1-5) and whether the review should read as
    positive or negative. Ratings 4-5 read positive, 1-2 read negative,
    3 is a coin flip since a 3-star review can go either way.
    """
    rating = random.choices([1, 2, 3, 4, 5], weights=[8, 10, 17, 30, 35])[0]
    if rating >= 4:
        sentiment = "positive"
    elif rating <= 2:
        sentiment = "negative"
    else:
        sentiment = random.choice(["positive", "negative"])
    return rating, sentiment


def random_date_last_two_months():
    """Pick a random date within the last ~60 days from today."""
    days_back = random.randint(0, 59)
    return datetime.now() - timedelta(days=days_back)


def make_one_response(index):
    rating, sentiment = pick_rating_and_sentiment()
    theme = random.choice(THEMES)
    sentence = random.choice(THEME_TEMPLATES[theme][sentiment])

    # Sometimes add a second sentence about a different theme, since real
    # reviews often mention more than one thing.
    if random.random() < 0.4:
        second_theme = random.choice(THEMES)
        second_rating, second_sentiment = pick_rating_and_sentiment()
        sentence += " " + random.choice(THEME_TEMPLATES[second_theme][second_sentiment])

    business = random.choice(BUSINESSES)
    survey = random.choice(SURVEYS)

    return {
        "response_id": f"r{index:06d}",
        "date": random_date_last_two_months().strftime("%Y-%m-%d"),
        "business_id": business["business_id"],
        "business_name": business["business_name"],
        "survey_id": survey["survey_id"],
        "survey_name": survey["survey_name"],
        "rating": rating,
        "response_channel": random.choice(CHANNELS),
        "free_text": sentence,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fake survey data for MiniSense")
    parser.add_argument("--count", type=int, default=8000, help="how many responses to generate")
    parser.add_argument("--out", type=str, default="data/survey_data.json", help="output file path")
    args = parser.parse_args()

    responses = [make_one_response(i) for i in range(1, args.count + 1)]

    with open(args.out, "w") as f:
        json.dump({"responses": responses}, f, indent=2)

    print(f"Wrote {len(responses)} fake survey responses to {args.out}")


if __name__ == "__main__":
    main()
