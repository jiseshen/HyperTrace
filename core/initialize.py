from .hypothesis_set import Hypothesis, WorkingBelief
from data import Turn
from .utils import TracerContext
from typing import List, Optional
import json


INITIALIZATION_PROMPT = """
You are initializing user preference hypotheses for a personalization pipeline. These hypotheses will guide future generation. Focus only on stable, evidence-supported signals from the most recent comparison between chosen and rejected responses.

Task

Given:
- Conversation history (may be empty for first turn)
- The latest user message
- Candidate responses (whether chosen or rejected specified)
- Previously retrieved hypotheses (may be empty)

Internally:
- Identify concrete differences between chosen and rejected responses.
- Determine which differences likely drove the user's choice.
- Only rely on differences clearly supported by evidence.

Possible dimensions (not exhaustive):
- Underlying values
- Information density
- Structure
- Level of abstraction
- Framing
- Actionability
- Tone

Then:

1. Decide whether to initialize a new belief state based on the given interaction.
   - If the user message contains insufficient information (e.g., simple greetings) or the candidate responses differ primarily in overall quality rather than stylistic or structural dimensions, such that no clear user-specific preference can be inferred, return "action": "skip" and produce no hypotheses.
2. Identify the category/topic of the current conversation.
3. Produce exactly {n_hypotheses} stable user preference hypotheses:
   - Reuse and revise relevant retrieved hypotheses when appropriate.
   - Otherwise generate new hypotheses.
   - Do not speculate beyond available evidence.
   - Do not force-fit dimensions.

Output Format

Return a JSON object:

{{
  "action": "initialize" | "skip",
  "category": "string",
  "hypotheses": [
    {{
      "id": "string (reuse existing ID or 'new-1')",
      "action": "reuse" | "new",
      "content": "string",
      "evidence": "short justification (optional)"
    }}
  ]
}}

Rules:
- Output valid JSON only.
- If the action is initialize, always include all required fields, and produce exactly {n_hypotheses} hypotheses.
- Ground each hypothesis in explicit evidence from the comparison.
- If skipping, leave other fields empty or null.

Conversation History:
{prev_turns}

Current Turn:
{current_turn}

Previously Retrieved Hypotheses:
{retrieved_hypotheses}
"""


def initialize_hypothesis(
    conversation_history: List[Turn],
    context: TracerContext
    ) -> Optional[WorkingBelief]:
    """Initialize a working belief with retrieved hypotheses. Return None if skipping this turn."""
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    candidate_hypotheses = context.hypothesis_set.retrieve_hypotheses(current_turn.user_message, top_k=context.tracer_config.n_hypotheses)
    
    prompt = INITIALIZATION_PROMPT.format(
        n_hypotheses=context.tracer_config.n_hypotheses,
        prev_turns="\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
        current_turn=current_turn.format(),
        retrieved_hypotheses="\n".join([h.format() for h in candidate_hypotheses])
    )
    
    retries = 0
    while True:
        retries += 1
        initialize_output = context.model.generate(prompt, cfg=context.generation_config)["output"]
        try:
            output_data = json.loads(initialize_output)
            new_hypotheses: List[Hypothesis] = []
            reused_hypotheses: List[Hypothesis] = []
            if output_data['action'] == 'skip':
                return None
            for h in output_data['hypotheses']:
                if h['action'] == 'reuse':
                    prev_category = context.hypothesis_set[h['id']][0].category
                    if output_data['category'] not in prev_category:
                        new_category = prev_category + ", " + output_data['category']
                    else:
                        new_category = prev_category
                    reused_hypotheses.append(Hypothesis(id=h['id'], category=new_category, content=h['content']))
                else:
                    new_hypotheses.append(Hypothesis(id=h['id'], category=output_data['category'], content=h['content']))
            if len(new_hypotheses) + len(reused_hypotheses) != context.tracer_config.n_hypotheses:
                continue
            break
        except:
            if retries >= context.generation_config.max_retries:
                raise ValueError(f"Failed to parse model output after {retries} attempts: {initialize_output}")
            
    context.hypothesis_set.update_hypotheses(reused_hypotheses)
    new_ids = context.hypothesis_set.add_hypotheses(new_hypotheses)
    reused_ids = [h.id for h in reused_hypotheses]
    all_ids = new_ids + reused_ids
    _, priors = context.hypothesis_set.retrieve_hypotheses(all_ids)
    belief = WorkingBelief(
        hypothesis_ids=all_ids,
        priors=priors
    )
    return belief