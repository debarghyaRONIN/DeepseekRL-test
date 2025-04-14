# Pokemon Red with Ollama deepseek-r1

This extension adds [Ollama](https://ollama.ai/) with the deepseek-r1 model to the Pokemon Red reinforcement learning agent. The integration allows the deepseek-r1 model to:

1. Assist with hyperparameter selection for the RL training
2. Provide strategic advice during gameplay
3. Analyze and summarize the agent's performance

## Requirements

- [Ollama](https://ollama.ai/) installed on your machine
- The deepseek-r1 model downloaded via Ollama
- Python 3.7+ with required packages (installed automatically by the launcher)

## Quick Start

The easiest way to get started is to run the launcher script:

```bash
python run_pokemon_deepseek.py
```

This will:
1. Check if Ollama is installed
2. Check if the deepseek-r1 model is available (and offer to download it if not)
3. Install required Python dependencies
4. Present you with options to train, play, or both

## How It Works

### Training with deepseek-r1 Guidance

When training a new model, the deepseek-r1 model provides advice on the optimal hyperparameters for the PPO algorithm based on its knowledge of reinforcement learning for game-playing agents.

```bash
python ollama_pokemon_agent.py --train
```

### Playing with deepseek-r1 Assistance

When playing Pokemon Red, the agent uses a hybrid approach:
- The trained RL policy makes most of the decisions
- Periodically (about 30% of the time), it consults the deepseek-r1 model for strategic advice

The deepseek-r1 model receives:
- Current game state information (HP, location, etc.)
- Recent rewards
- Recent actions taken

It then recommends the next action based on its expert knowledge of Pokemon Red and reinforcement learning.

```bash
python ollama_pokemon_agent.py --play
```

At the end of the playthrough, the deepseek-r1 model provides a summary of the agent's performance.

### Parameters

By default, the hybrid agent uses the deepseek-r1 model for approximately 30% of its decisions. You can adjust this in the code by changing the `use_frequency` parameter in the `OllamaAdvisor` class.

## Architecture

The integration has several key components:

1. **OllamaAdvisor**: Combines RL policy with deepseek-r1 advice
2. **query_ollama**: Handles communication with the Ollama API
3. **get_ollama_strategy**: Formats game state into prompts for deepseek-r1
4. **train_with_ollama_guidance**: Trains the RL agent with deepseek-r1 advice
5. **play_with_ollama_and_rl**: Plays Pokemon Red with the hybrid approach

## Customization

You can modify the prompts in the `get_ollama_strategy` function to provide more context to the deepseek-r1 model, potentially improving its advice.

You can also adjust hyperparameters in `train_with_ollama_guidance` or modify the environment configuration as needed.

## Troubleshooting

If you encounter issues:

1. Ensure Ollama is running (`ollama serve`)
2. Check if deepseek-r1 is installed (`ollama list`)
3. If not installed, download it with `ollama pull deepseek-r1`
4. Ensure the Pokemon ROM and initial state files are in the correct locations
5. Check the API responses from Ollama by enabling debug logging

## Extending

This integration could be extended to:
- Use different Ollama models
- Provide more detailed game state information to the LLM
- Implement more sophisticated hybrid decision-making approaches
- Add real-time visualization of the LLM's reasoning 