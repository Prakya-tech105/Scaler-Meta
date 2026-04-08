# Hugging Face Spaces Notes

This project is ready to run on Hugging Face Spaces as a Python Gradio app.

## Recommended setup
- Use the `app.py` file at the repository root as the Space entrypoint.
- Keep `requirements.txt` at the root so Spaces can install dependencies.
- Ensure the Space runtime can access `src/` by using the `app.py` path bootstrap.

## Launch behavior
- The app listens on `0.0.0.0:7860`.
- The UI will fall back to the rule-based baseline if no DQN model is available.

## Suggested Space files
- `app.py`
- `requirements.txt`
- `src/qwt_optimizer/...`
- `outputs/models/` for an optional trained model artifact

## If you deploy a trained model
- Upload the saved SB3 model zip to `outputs/models/` or update the model path in the UI.
- If the file is missing, the UI still works with the baseline policy.

## Important assumption
- The project uses a practical Gymnasium + SB3 implementation with an OpenEnv dependency listed in `requirements.txt`. If your OpenEnv distribution uses a different pip package name, update `requirements.txt` accordingly before deployment.
