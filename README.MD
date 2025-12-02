# XPIA Defense Dataset

This dataset supports research into defenses against Indirect Prompt
Injection Attacks (XPIA) in agentic browsing pipelines.

## Datasets

### spotlight_xpia_test.json
- 1000 injected documents
- Tasks: summarization, Q&A
- Random payload placement (start/middle/end)
- Payload verification keyword: "HACKED"

### jatmo_train.json
- Synthetic fine-tuning pairs
- GPT-style teacher labelling
- Tasks: summarization, translation

### jatmo_attack_grid.json
- Testing matrix for prompt injection attacks
- Multiple injection formulations × multiple locations

## Ethics
All data is synthetic or public-domain.
No personal or copyrighted data included.

## License
Creative Commons Attribution 4.0 (CC-BY 4.0)
