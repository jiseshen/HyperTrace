# X thread draft

**Post 1 — head graphic**

Can your agent smoothly adapt to your preferences across sessions?

Meet HyperTrace, a training-free LLM personalization framework. It tracks hypotheses about what you want now and what tends to last, rather than reducing you to a fixed profile or a list of memories. 🧵

[Upload: head graphic]

**Post 2 — main workflow graphic**

At each turn, HyperTrace reweights intent and preference hypotheses with an SMC-style score of the user's choice. Topic-aware retrieval brings relevant long-term beliefs into a new session; evidence-weighted consolidation strengthens what repeatedly holds.

[Upload: main workflow graphic]

**Post 3 — results graphic**

On PRISM, HyperTrace stays steadier through a session switch. After turn 20, choice-prediction accuracy is 61.4% vs 54.4% for a memory-based method. It also leads with 80% of feedback withheld and frequent topic shifts—stability even when evidence is thin.

[Upload: results graphic]

**Post 4 — closing and links**

Personalization must distinguish a lasting preference from today's intent. HyperTrace tracks both over time, staying robust as topics change without updating model parameters.

Paper: https://arxiv.org/abs/2609.09835
Code: https://github.com/jiseshen/HyperTrace
