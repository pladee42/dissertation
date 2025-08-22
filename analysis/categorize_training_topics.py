#!/usr/bin/env python3
"""
Script to categorize training topics T0001-T0100 using keyword matching.
This ensures accurate representation of topic distribution in methodology.tex.
"""

import json
from typing import Dict, List, Set
from collections import Counter

def load_training_topics(topics_file: str) -> Dict[str, str]:
    """Load training topics from JSON file."""
    with open(topics_file, 'r') as f:
        topics = json.load(f)
    
    # Create mapping from UID to topic name
    topic_mapping = {}
    for topic in topics:
        topic_mapping[topic['uid']] = topic['topic_name']
    
    return topic_mapping

def categorize_training_topics(topics_file: str) -> Dict[str, str]:
    """Categorize training topics T0001-T0100 using keyword matching."""
    
    # Load topics
    topics = load_training_topics(topics_file)
    
    # Define category keywords (same as validation topics)
    categories = {
        "Environmental/Wildlife Conservation": [
            "wildlife", "endangered", "species", "polar bear", "whale", "conservation", 
            "environmental", "animal", "forest", "rainforest", "deforestation", "elephant", 
            "poaching", "pesticide", "bee", "turtle", "rehabilitation", "shelter animals",
            "factory farming", "wildfire", "coral reef", "ocean acidification", "plastic pollution",
            "wolf", "cosmetic testing", "migratory bird", "overfishing", "shark", "preserve"
        ],
        "Human Rights/Social Justice": [
            "human rights", "social justice", "lgbtq", "equality", "pride", "transgender", 
            "discrimination", "amnesty", "activist", "prisoner", "justice", "rights", 
            "hrc", "inclusive", "workplace", "refugee", "child soldiers", "political prisoner",
            "censorship", "women's rights", "gender equality", "indigenous", "forced labor",
            "death penalty", "medical supplies", "conflict", "election", "observer",
            "belarusian", "xinjiang", "uae", "mansoor"
        ],
        "Community/Charity Support": [
            "children", "youth", "community", "charity", "fundraising", "volunteer", 
            "donation", "support", "kids", "school", "mentorship", "center", "toy drive",
            "summer camp", "soup kitchen", "financial literacy", "food drive", "coat",
            "homeless", "graduation", "facility dog", "gala", "auction", "aftercare",
            "back-to-school", "tutoring", "sports equipment", "open house"
        ],
        "Media/Broadcasting/News": [
            "radio", "podcast", "news", "broadcasting", "media", "whyy", "show", "program",
            "journalism", "documentary", "tv", "newsletter", "investigative", "storytelling",
            "mobile app", "streaming", "programming", "black history month"
        ]
    }
    
    # Categorize topics
    topic_categories = {}
    
    for topic_uid, topic_name in topics.items():
        topic_lower = topic_name.lower()
        categorized = False
        
        # Try to match keywords
        for category, keywords in categories.items():
            if any(keyword.lower() in topic_lower for keyword in keywords):
                topic_categories[topic_uid] = category
                categorized = True
                break
        
        # Manual assignment for specific topics that might not match keywords
        if not categorized:
            # These are based on analysis of the topic content
            manual_assignments = {
                "T0024": "Media/Broadcasting/News",      # Water scarcity coverage
                "T0044": "Human Rights/Social Justice",  # Don't Say Gay legislation
                "T0048": "Human Rights/Social Justice",  # It Gets Better Project
                "T0052": "Human Rights/Social Justice",  # Chapter annual meeting
                "T0053": "Human Rights/Social Justice",  # Year-end advocacy report
                "T0099": "Media/Broadcasting/News",      # Meet-and-greet with hosts
            }
            
            if topic_uid in manual_assignments:
                topic_categories[topic_uid] = manual_assignments[topic_uid]
                categorized = True
        
        if not categorized:
            print(f"Warning: Could not categorize {topic_uid}: {topic_name}")
    
    return topic_categories

def analyze_distribution(topic_categories: Dict[str, str]) -> Dict[str, int]:
    """Analyze the distribution of topics across categories."""
    category_counts = Counter(topic_categories.values())
    return dict(category_counts)

def main():
    """Main function to categorize training topics and generate report."""
    topics_file = "config/topics.json"
    
    print("Categorizing training topics T0001-T0100...")
    
    # Categorize topics
    topic_categories = categorize_training_topics(topics_file)
    
    # Analyze distribution
    category_counts = analyze_distribution(topic_categories)
    
    # Generate report
    print("\n=== Training Topic Categorization Results ===")
    print(f"Total topics categorized: {len(topic_categories)}")
    print("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} topics")
    
    # Save detailed results
    results = {
        "topic_categories": topic_categories,
        "category_counts": category_counts,
        "summary": {
            "total_topics": len(topic_categories),
            "categories": list(category_counts.keys())
        }
    }
    
    output_file = "analysis/training_topics_categorized.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Show topics by category for verification
    print("\n=== Topics by Category ===")
    topics = load_training_topics(topics_file)
    
    for category in sorted(category_counts.keys()):
        print(f"\n{category} ({category_counts[category]} topics):")
        category_topics = [uid for uid, cat in topic_categories.items() if cat == category]
        for uid in sorted(category_topics):
            print(f"  {uid}: {topics[uid]}")

if __name__ == "__main__":
    main()