# PromptForge

A ComfyUI custom node pack for intelligent prompt engineering.

## Features

- **Tag Tagger** — automatically extract and suggest quality/style tags for prompts.
- **Prompt Expander** — enrich a short prompt into a detailed description via LLM.
- **Translator** — translate prompts between languages while preserving artistic terms.
- **Vision Analyser** — use a vision LLM to analyse images and produce matching prompts.
- **Video Prompter** — generate frame-by-frame or motion-style prompts for video workflows.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-org>/PromptForge.git
pip install -r PromptForge/requirements.txt
```

## Project Structure

```
PromptForge/
├── __init__.py            # ComfyUI node registration
├── requirements.txt       # Python dependencies
├── config/
│   ├── tags/              # Tag vocabulary / mappings
│   ├── rules/
│   │   ├── expand/        # Expansion rule templates
│   │   ├── translate/     # Translation glossaries
│   │   ├── vision/        # Vision analysis prompts
│   │   └── video/         # Video prompt templates
│   └── presets/           # Saved preset configurations
└── README.md
```

## Roadmap

- [ ] Implement Tag Tagger node
- [ ] Implement Prompt Expander node
- [ ] Implement Translator node
- [ ] Implement Vision Analyser node
- [ ] Implement Video Prompter node
- [ ] Add preset import / export UI

## License

MIT
