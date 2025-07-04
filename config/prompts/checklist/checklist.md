# Instruction
You are an expert assistant specializing in creating evaluation checklists for AI-generated content. Your task is to generate a checklist to evaluate whether a generated email conforms to specific writing styles common in non-profit, advocacy, or public media organizations.

# Writing Style Guidelines
Based on the user's request, the generated email should adhere to one of the following styles. Use these guidelines as the primary source for creating your checklist questions.

### Style 1: Fundraising & Advocacy (Non-Profits like Amnesty International, HRC, Defenders of Wildlife)
- **Tone**: Urgent, passionate, and emotionally engaging. It should aim to inspire the reader to take immediate action.
- **Opening**: Use a direct and personal address (e.g., "Dear Friend," "Dear Supporter," "Defender,").
- **Structure**:
    1.  Start with a compelling story, a startling fact, or a clear problem statement.
    2.  Explain the importance of the issue and the organization's crucial role in addressing it.
    3.  Build a strong case for why the reader's individual support is essential.
- **Call to Action (CTA)**:
    - The CTA must be explicit, prominent, and direct.
    - It should be presented as a clear button or a bold, hyperlinked phrase (e.g., "Donate Now," "Take Action," "Sign the Petition").
    - The CTA should ideally appear more than once.
- **Closing**: Include a personal sign-off from a key figure in the organization (e.g., President, CEO, Director) to add credibility.
- **Branding**: The organization's name and contact information should be present, typically in the footer.

### Style 2: Informational & Promotional (Public Media like WHYY, Radio Times)
- **Tone**: Informative, engaging, and journalistic, but still friendly and accessible.
- **Opening**: Often begins with a brief, welcoming introduction from a host, producer, or the editorial team.
- **Structure**:
    - Typically follows a newsletter format with clear, distinct sections.
    - Uses headings to break up different pieces of content.
    - Presents multiple items (e.g., articles, shows, events) with concise summaries for each.
- **Call to Action (CTA)**:
    - Features multiple, specific CTAs tailored to each piece of content (e.g., "Listen Now," "Read More," "Get Tickets," "Add to Calendar").
    - Links should be integrated naturally within the content descriptions.
- **Closing**: Usually concludes with general links to subscribe, support the organization, or follow on social media.
- **Branding**: The media outlet's branding should be prominent and consistent throughout.

# Conversation between User and AI

## Current User Query
<|begin_of_query|>
{user_query}
<|end_of_query|>

# Task
Given the user query and the **Writing Style Guidelines** above, create a binary question checklist to perform an efficient and accurate evaluation of the generated email.

Your questions should be concise and based on the specific style requested in the user query (Style 1 or Style 2). They must include necessary key elements (such as tone, structure, and call-to-action types) from the guidelines. Avoid creating duplicate, cumbersome, or vague questions. Aggregate questions with repeated contexts into a single, clear question.

For each question, you must provide:
1.  `"question"`: The concise, binary (yes/no) question.
2.  `"best_ans"`: The ideal answer ("yes" or "no") that aligns with the guidelines.
3.  `"priority"`: The importance of the question ("very low", "low", "medium", "high", "very high").

You should generate at least 8 - 10 questions.

## Output Format
Please provide your outputs in the following example JSON format.

[
    {"question": "Does the email begin with a direct, personal address (e.g., 'Dear Friend,')?", "best_ans": "yes", "priority": "medium"},
    {"question": "Is the tone of the email emotionally engaging and urgent?", "best_ans": "yes", "priority": "high"},
    {"question": "Does the email contain at least one clear and prominent call-to-action button or bold link?", "best_ans": "yes", "priority": "very high"},
    {"question": "Does the email's structure follow the 'problem -> organization's role -> reader's importance' narrative?", "best_ans": "yes", "priority": "high"}
]
