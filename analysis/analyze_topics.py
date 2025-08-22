import json

def analyze_topics(file_path):
    categories = {
        "Environmental/Wildlife Conservation": [
            "polar bear", "wildlife", "endangered species", "whale", "amazon rainforest",
            "elephant poaching", "pesticides", "bee populations", "sea turtle",
            "wildlife rehabilitation", "animal welfare", "australian wildfires",
            "coral reefs", "plastic pollution", "marine life", "gray wolf",
            "cosmetic testing", "migratory bird", "overfishing", "shark populations",
            "wildlife preserve", "water scarcity", "environmental issues", "logging", "congo basin",
            "vaquita porpoise", "world oceans day", "plastic footprint", "reforestation",
            "climate change", "extreme weather", "marine protected area", "river cleanup",
            "orangutans", "earth day", "forest fire"
        ],
        "Human Rights/Social Justice": [
            "human rights", "LGBTQ+", "equality act", "transgender", "inclusive workplaces",
            "pride", "legal aid", "mental health resources", "justice", "refugee crisis",
            "child soldiers", "wrongfully imprisoned", "internet censorship", "women's rights",
            "indigenous land rights", "persecuted human rights activists",
            "climate change on human rights", "international human rights treaty",
            "forced labor", "death penalty", "activist", "hrc", "lgbtq", "ally", "supreme court victory", "election observers",
            "amnesty international", "prisoner", "belarusian case", "legislation", "it gets better project", "election", "it gets better",
            "safe schools improvement act", "national equality gala", "workplace", "canvass",
            "crisis hotline", "bisexual community", "discriminatory bills", "marriage equality",
            "senior healthcare", "activists around the world", "war crimes", "jailed journalists",
            "emergency aid", "displaced families", "facial recognition technology", "dissent",
            "corporate complicity", "clean water as a human right", "child marriage", "the hague",
            "senator", "letter-writing campaign", "yemen", "myanmar", "sudan", "suppress dissent", "clean water", "accountability",
            "safe schools", "discriminatory bills", "marriage equality", "war crimes", "jailed journalists", "emergency aid", "displaced families", "facial recognition technology", "suppress dissent", "clean water",
            "safe schools improvement act", "letter-writing campaign", "discriminatory bills", "marriage equality",
            "war crimes", "jailed journalists", "emergency aid", "displaced families", "facial recognition technology", "suppress dissent", "clean water"
        ],
        "Community/Charity Support": [
            "giving tuesday", "fundraising", "membership renewal", "volunteer", "children's home",
            "march for kids", "gala", "charitable giving", "aftercare program", "food drive",
            "soup kitchen", "toy drive", "scholarships", "community sports", "open house",
            "back-to-school", "mentorship", "tutors", "homeless youth", "shelter animals",
            "community", "financial literacy", "graduation ceremony", "food bank", "medical supplies",
            "general meeting", "advocacy achievements", "youth community", "foster parents",
            "school supplies", "champions for children", "mobile library", "read to children",
            "family in crisis", "musical instruments", "arts program", "cyberbullying",
            "playground", "boys & girls club", "fund drive", "parents", "hospital"
        ],
        "Media/Broadcasting/News": [
            "radio times", "whyy", "news", "podcast", "talk show", "documentary",
            "listener survey", "program highlights", "mobile app", "storytelling project",
            "journalism", "station hosts", "black history month", "tv guide", "live show",
            "radio drama", "educational animated kids' show", "science friday", "hiring",
            "pitches", "public radio host", "tote bag", "misinformation", "election night analysis",
            "loyal listener"
        ]
    }

    category_counts = {cat: 0 for cat in categories}
    uncategorized_topics = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            topic_name = item.get("topic_name", "").lower()
            assigned_category = None
            for category, keywords in categories.items():
                if any(keyword in topic_name for keyword in keywords):
                    category_counts[category] += 1
                    assigned_category = category
                    break
            if assigned_category is None:
                uncategorized_topics.append(topic_name)

        print("Category Counts:")
        for category, count in category_counts.items():
            print(f"- {category}: {count}")
        
        if uncategorized_topics:
            print("\nUncategorized Topics:")
            for topic in uncategorized_topics:
                print(f"- {topic}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    analyze_topics("/Users/tan_waris/Library/CloudStorage/GoogleDrive-wratthapoom1@sheffield.ac.uk/My Drive/Dissertation/email_samples/topic_val.json")