from hypothesis_set import Hypothesis, HypothesisSet, WorkingBelief, Update
from .utils import TracerContext
from data import Turn
from typing import List, Optional
import json
from .initialize import initialize_hypothesis


BRANCHING_PROMPT = """
You are updating ONE working hypothesis in a personalization pipeline.

You are given:
- The current hypothesis (category + content)
- The latest user interaction: user message + candidate responses (chosen vs rejected)
- (Optional) brief conversation history from previous turns

Goal:
Decide whether this interaction provides usable preference evidence for updating this hypothesis, and if so, how.

Step 1: Evidence Gate
Quickly decide whether this interaction provides stable, user-specific preference evidence.

If:
- The user message is trivial (e.g., greetings), OR
- The chosen vs rejected difference is primarily overall quality (e.g., correctness or completeness) rather than stylistic/structural/preference-based differences,
AND
- The previous context provides no explicit or stable preference signals,

Then:
- set action="skip" and return directly.

Step 2: Relevance (if not skip)
Assess whether the interaction is relevant to this hypothesis:
- relevance="direct" if the interaction directly supports or contradicts this hypothesis.
- relevance="partial" if topic differs but the same underlying preference/value/style is evidenced.
- relevance="none" if no meaningful support exists.

Step 3: Update (only if not skip)
Internally (do NOT output this analysis):
- Compare chosen vs rejected to extract preference-relevant differences.
- Consider possible dimensions (do not force-fit):
  * Underlying values
  * Information density
  * Structure
  * Level of abstraction
  * Framing
  * Actionability
  * Tone
- Scan previous context (especially follow-ups) for:
  * Explicit corrections
  * Stated preferences
  * Constraints
  * Dissatisfaction signals
  * Consistent behavioral patterns
- Do NOT infer latent motivations without direct evidence.
- Treat previous context as non-evidence if no stable signal is present.

Update decision:
- If relevance is "direct" or "partial":
    action="revise"
    Produce an updated hypothesis that:
      * Incorporates new evidence,
      * Reduces inconsistency,
      * Remains specific (do not over-generalize),
      * Keeps the original intent unless clearly contradicted.
- If relevance is "none":
    action="replace"
    Produce a new hypothesis at similar specificity.

Category update rules:
- If the current turn provides clear evidence that the topic category has shifted, update the category accordingly.
- Otherwise, keep the original category.
- If the original category contains multiple labels, select the single most appropriate one based on the current evidence.
- Do not introduce a new category unless clearly justified by the interaction.

Output JSON only:

{{
  "action": "skip" | "revise" | "replace",
  "relevance": "direct" | "partial" | "none",
  "updated_hypothesis": {{
    "category": "string",
    "content": "string"
  }} | null,
  "evidence": "one short justification"
}}

Rules:
- If action="skip": updated_hypothesis must be null.
- If action="revise": category should usually remain the same.
- If action="replace": do not reference the old hypothesis in the new content.
- Ground all updates strictly in observed evidence.
- Avoid speculation and over-generalization.

Conversation History:
{prev_turns}

Current Turn:
{current_turn}

Note:
The user message may provide preference evidence only if it introduces explicit constraints, corrections, refinements, or consistent behavioral signals. Otherwise, do not treat it as preference evidence.

Current Hypothesis:
{current_hypothesis}
"""


def branch_hypotheses(conversation_history: List[Turn], context: TracerContext) -> Optional[WorkingBelief]:
    """Propagate hypotheses based on new user message. Skip if no usable evidence, reinitialize if irrelevant, or simply revise."""
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    current_hypotheses = context.belief.get_hypotheses()
    prompts = [
        BRANCHING_PROMPT.format(
            prev_turns="\n\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
            current_turn=current_turn.format(include_candidates=True),
            current_hypothesis=h.format()
        ) for h in current_hypotheses
    ]
    outputs = [o["output"] for o in context.model.async_generate(prompts, cfg=context.generation_config)]
    
    skip, replace, invalid = 0, 0, 0
    for output in outputs:
        try:
            output_data = json.loads(output)
            if output_data['action'] == 'skip':
                skip += 1
            elif output_data['action'] == 'replace':
                replace += 1
        except:
            invalid += 1
    if skip > len(outputs) / 2:  # Vote to skip
        return None
    if replace > len(outputs) / 2:  # Vote to reinitialize
        context.belief.consolidate(context.tracer_config.importance, context.tracer_config.consolidate_alpha)
        return initialize_hypothesis(conversation_history, context)
    if replace > 0:
        context.belief.consolidate(context.tracer_config.importance, context.tracer_config.consolidate_alpha)
    updates = []
    revised_ids = []
    new = []
    for h, o in zip(current_hypotheses, outputs):
        if o['action'] == 'revise':
            new_cat = o['updated_hypothesis']['category']
            new_cat = h.category if new_cat in h.category else h.category + ", " + new_cat
            updates.append(Update(h.id, category=new_cat, content=o['updated_hypothesis']['content']))
            revised_ids.append(h.id)
        elif o['action'] == 'replace':
            new.append(o['updated_hypothesis'])
    context.hypothesis_set.update_hypotheses(updates)
    new_ids = context.hypothesis_set.add_hypotheses(new)
    all_ids = revised_ids + new_ids
    _, priors = context.hypothesis_set[all_ids]
    return WorkingBelief(ids=all_ids, priors=priors, repo=context.hypothesis_set)
    