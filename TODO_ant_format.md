# Plan for Changing Model Save/Load to .ant Format

## Information Gathered
- Current format: Pickle (.pkl files)
- New format: Custom .ant format (JSON with base64-encoded numpy arrays)
- Main save/load in src/core/graph.py (AntGraph.save/load)
- Called from src/core/engine.py (save_checkpoint)
- Filenames in main.py: 'final_model.pkl' -> 'final_model.ant', 'interrupted_model.pkl' -> 'interrupted_model.ant'
- Also in examples/example_usage.py: 'examples/vocab_example.json' but for model it's .pkl

## Plan
1. [x] Modify src/core/graph.py:
   - [x] Add imports: import json, base64
   - [x] Change save method: Convert numpy arrays to base64, save as JSON
   - [x] Change load method: Load JSON, decode base64 back to numpy arrays
   - [x] Change file extension to .ant in print messages

2. [x] Update src/core/engine.py:
   - [x] Change checkpoint filename from .pkl to .ant
   - [x] Change stats_path replace from '.pkl' to '.ant'
   - [x] Change get_latest_checkpoint to look for .ant files

3. [x] Update main.py:
   - [x] Change 'final_model.pkl' to 'final_model.ant'
   - [x] Change 'interrupted_model.pkl' to 'interrupted_model.ant'
   - [x] Change chat_mode to look for .ant files

4. [x] Update examples/example_usage.py:
   - [x] Change model load to look for .ant files

5. [x] Update any documentation or config if needed

## Dependent Files
- [x] src/core/graph.py (main change)
- [x] src/core/engine.py
- [x] main.py
- [x] examples/example_usage.py

## Followup Steps
- Test saving and loading works
- Verify backward compatibility (old .pkl files can't be loaded with new method)
- Update any tests if needed
