flowchart TD
    X["Example: question + reference answer"]
    X -.->|"format and tokenize"| P["Prompt token IDs"]
    P -->|"collect_completions()"| C["Completion: token_ids + finish_reason"]
    C -.->|"verify reward; attach prompt"| E["Episode: prompt_ids + response_ids + reward"]

    E -->|"estimate_vineppo(); supplied step boundaries"| V["VinePPOValueEstimate: episode + boundaries + MC values + step advantages"]
    V -->|"vineppo_advantage_method()"| A["list[float]: one advantage per response token"]
    A -.->|"wrap with original Episode fields"| EA["EpisodeWithAdvantages"]

    E -.->|"critic values + GAE; not implemented"| Q["PPOValueEstimate: episode + cached values + return targets"]
    Q -.->|"derive token advantages; not implemented"| EA
    Q -.->|"collate_critic(); stub"| CB["CriticBatch"]

    EA -->|"collate(); supply old and reference logprobs"| PB["PolicyGradientBatch"]
    PB -.->|"update_policy(); not implemented"| U["Actor update"]
    CB -.->|"not implemented"| CU["Critic update"]
