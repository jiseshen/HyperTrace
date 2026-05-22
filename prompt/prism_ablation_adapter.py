from dataclasses import replace

from .base import _insert_after
from .prism_adapter import prism_prompts


FLAT5_BRANCHING_INSERT = """
Flat-5 ablation mode:
- The system keeps exactly the current five hypothesis slots for the user.
- If relevance is "none", action="replace" rewrites this same slot in place with a new hypothesis from the current evidence.
- Do not assume a separate retrieval, consolidation, or reinitialization step will repair unrelated hypotheses later.
- Keep replacements specific and evidence-grounded; do not mention the old hypothesis content.
"""


def prism_flat5_ablation_prompts():
    prompts = prism_prompts()
    return replace(
        prompts,
        branching=_insert_after(
            prompts.branching,
            "- Avoid speculation and over-generalization.",
            FLAT5_BRANCHING_INSERT,
        ),
    )
