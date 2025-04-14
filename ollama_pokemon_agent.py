import sys
import json
import time
import subprocess
import requests
import os
import glob
from os.path import exists
from pathlib import Path
import numpy as np

# Print conda environment info
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
print(f"Using conda environment: {conda_env or 'None'}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Try imports with diagnostic output
try:
    print("Importing RedGymEnv...")
    from red_gym_env_v2 import RedGymEnv
    print("Importing StreamWrapper...")
    from stream_agent_wrapper import StreamWrapper
    print("Importing PPO...")
    from stable_baselines3 import PPO
    print("Importing env_checker...")
    from stable_baselines3.common import env_checker
    print("Importing SubprocVecEnv...")
    from stable_baselines3.common.vec_env import SubprocVecEnv
    print("Importing set_random_seed...")
    from stable_baselines3.common.utils import set_random_seed
    print("Importing callbacks...")
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    print("Importing TensorboardCallback...")
    from tensorboard_callback import TensorboardCallback
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# Ollama API interaction
def query_ollama(prompt, model="deepseek-r1"):
    """Query the Ollama API with the given prompt."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

def get_ollama_strategy(game_state, observation, reward_info, recent_actions):
    """Get strategic advice from the Ollama deepseek-r1 model."""
    prompt = f"""
You are an expert Pokemon Red player and reinforcement learning agent. 
You need to provide strategic advice for the next action to take in the game.

Current game state:
- HP: {game_state.get('hp', 'Unknown')}
- Location: {game_state.get('location', 'Unknown')}
- Current Pokemon: {game_state.get('current_pokemon', 'Unknown')}
- Badges: {game_state.get('badges', [])}
- Items: {game_state.get('items', [])}

Recent rewards:
{json.dumps(reward_info, indent=2)}

Recent actions taken:
{recent_actions}

Based on this information, what should be the next action? Choose from:
0: Right
1: A
2: Left
3: Down
4: Up
5: B
6: Start
7: Select

Provide your recommendation as a JSON object with the following format:
{{
  "action": <action_number>,
  "reasoning": "<your reasoning>",
  "exploration_strategy": "<exploration or exploitation>"
}}
"""
    response = query_ollama(prompt)
    try:
        # Try to parse the response as JSON
        result = json.loads(response)
        return result
    except:
        # If it's not valid JSON, try to extract just the action number
        try:
            import re
            action_match = re.search(r'"action":\s*(\d)', response)
            if action_match:
                action = int(action_match.group(1))
                return {"action": action, "reasoning": "Extracted from response", "exploration_strategy": "unknown"}
        except:
            pass
    
    # Default fallback
    return {"action": 1, "reasoning": "Default action (A button)", "exploration_strategy": "default"}

def make_env_with_ollama(rank, env_conf, seed=0):
    """Create an environment that leverages Ollama for strategic decisions."""
    def _init():
        base_env = RedGymEnv(env_conf)
        env = StreamWrapper(
            base_env, 
            stream_metadata = {
                "user": "deepseek-agent",
                "env_id": rank,
                "color": "#447799",
                "extra": "Powered by deepseek-r1",
            }
        )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

class OllamaAdvisor:
    """Class that integrates Ollama deepseek-r1 advice with the RL agent."""
    def __init__(self, model, env, use_frequency=0.3):
        self.model = model
        self.env = env
        self.use_frequency = use_frequency  # How often to use Ollama advice vs RL policy
        self.advice_history = []
        
    def get_action(self, observation):
        """Get action to take, either from RL model or Ollama."""
        # Get the RL model's action
        action, _states = self.model.predict(observation, deterministic=False)
        
        # Sometimes use Ollama's advice
        if np.random.random() < self.use_frequency:
            try:
                # Get game state information
                if hasattr(self.env, 'get_game_state_reward'):
                    reward_info = self.env.get_game_state_reward()
                else:
                    # Try to get it from the first env in case it's a vec env
                    if hasattr(self.env, 'envs') and hasattr(self.env.envs[0], 'get_game_state_reward'):
                        reward_info = self.env.envs[0].get_game_state_reward()
                    else:
                        reward_info = {}
                
                # Get more state information if possible
                game_state = {}
                if hasattr(self.env, 'unwrapped'):
                    game_env = self.env.unwrapped
                    if hasattr(game_env, 'read_hp_fraction'):
                        game_state['hp'] = game_env.read_hp_fraction()
                    # Add more state information as available
                
                # Get recent actions if available
                if hasattr(self.env, 'recent_actions'):
                    recent_actions = self.env.recent_actions.tolist()
                else:
                    recent_actions = []
                
                # Query Ollama for advice
                advice = get_ollama_strategy(game_state, observation, reward_info, recent_actions)
                self.advice_history.append(advice)
                
                # Use the advised action
                if 'action' in advice and isinstance(advice['action'], int) and 0 <= advice['action'] < 8:
                    print(f"🤖 Ollama suggests: {advice['action']} - {advice['reasoning']}")
                    return advice['action']
                
            except Exception as e:
                print(f"Error getting Ollama advice: {e}")
        
        return action

def train_with_ollama_guidance():
    """Train the Pokemon agent with assistance from Ollama deepseek-r1."""
    # Initialize Ollama if not already running
    try:
        subprocess.run(["ollama", "run", "deepseek-r1", "--nointeract", "--prompt", "Hello"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except:
        print("Make sure Ollama is installed and the deepseek-r1 model is available")
        print("You can install it with: ollama pull deepseek-r1")
        return
    
    # Set up training environment
    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_id = "deepseek_runs"
    sess_path = Path(sess_id)
    sess_path.mkdir(exist_ok=True)
    
    env_config = {
        'headless': False, 'save_final_state': False, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25
    }
    
    print(env_config)
    
    # Get deepseek-r1 advice on the training hyperparameters
    hyperparameter_prompt = """
As an expert in reinforcement learning for game-playing agents, suggest optimal hyperparameters 
for training a PPO agent to play Pokemon Red. Consider:

- Learning rate
- Batch size
- GAE lambda value
- Gamma (discount factor)
- Entropy coefficient
- Number of epochs per training batch

Provide your recommendations as a JSON object with numeric values.
"""
    
    try:
        hyperparams_response = query_ollama(hyperparameter_prompt)
        hyperparams = json.loads(hyperparams_response)
        print("Using deepseek-recommended hyperparameters:", hyperparams)
    except:
        # Fallback to default hyperparameters
        hyperparams = {
            "learning_rate": 3e-4,
            "batch_size": 1024,
            "gae_lambda": 0.95,
            "gamma": 0.999,
            "ent_coef": 0.02,
            "n_epochs": 3
        }
        print("Using default hyperparameters:", hyperparams)
    
    # Create environments
    num_cpu = 12
    env = SubprocVecEnv([make_env_with_ollama(i, env_config) for i in range(num_cpu)])
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="deepseek_poke")
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]
    
    # Set up WandB logging if enabled
    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train-deepseek",
            id=sess_id,
            name="deepseek-r1-agent",
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())
    
    # Check for existing model checkpoint
    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip()
    
    train_steps_batch = ep_length // 64
    
    if exists(file_name + ".zip"):
        print("\nLoading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        # Create new model with hyperparameters partially guided by deepseek-r1
        model = PPO(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            n_steps=train_steps_batch, 
            batch_size=hyperparams["batch_size"], 
            n_epochs=hyperparams["n_epochs"], 
            gamma=hyperparams["gamma"], 
            gae_lambda=hyperparams["gae_lambda"],
            ent_coef=hyperparams["ent_coef"],
            learning_rate=hyperparams["learning_rate"],
            tensorboard_log=sess_path
        )
    
    print(model.policy)
    
    # Train the model
    model.learn(total_timesteps=(ep_length)*num_cpu*10000, callback=CallbackList(callbacks), tb_log_name="deepseek_ppo")
    
    if use_wandb_logging:
        run.finish()
    
    return model, env

def play_with_ollama_and_rl(model_path=None):
    """Play Pokemon with a combination of RL model and Ollama deepseek-r1 advice."""
    # Set up environment for playing
    sess_path = Path("deepseek_play_session")
    sess_path.mkdir(exist_ok=True)
    
    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': 2**23, 
        'print_rewards': True, 'save_video': True, 'fast_video': False, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': True, 'reward_scale': 0.5, 'explore_weight': 0.25
    }
    
    # Create environment
    env = make_env_with_ollama(0, env_config)()
    
    # Load model
    if model_path is None:
        # Find most recent model
        model_files = glob.glob("deepseek_runs/deepseek_poke_*.zip")
        if model_files:
            model_path = max(model_files, key=os.path.getctime)
        else:
            print("No model found. Training a new one...")
            model, _ = train_with_ollama_guidance()
            return
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    # Create advisor that combines model and Ollama
    advisor = OllamaAdvisor(model, env, use_frequency=0.3)
    
    # Play the game
    obs, info = env.reset()
    done = False
    
    print("Starting game with deepseek-r1 assistance")
    
    while not done:
        # Get action from the advisor (combines RL and Ollama)
        action = advisor.get_action(obs)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # Check if done
        done = terminated or truncated
    
    env.close()
    
    # Summarize the playthrough with deepseek-r1
    summary_prompt = """
Summarize the Pokemon Red playthrough based on the following advice history:
""" + json.dumps(advisor.advice_history, indent=2) + """

Provide insights on:
1. Key strategies used
2. Areas where the agent performed well
3. Areas for improvement
4. Overall assessment of the agent's gameplay
"""
    
    summary = query_ollama(summary_prompt)
    print("\n===== DEEPSEEK-R1 PLAYTHROUGH SUMMARY =====")
    print(summary)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pokemon Red with Ollama deepseek-r1")
    parser.add_argument("--train", action="store_true", help="Train a new model with deepseek-r1 guidance")
    parser.add_argument("--play", action="store_true", help="Play with an existing model and deepseek-r1")
    parser.add_argument("--model", type=str, help="Path to model for playing", default=None)
    
    args = parser.parse_args()
    
    if args.train:
        train_with_ollama_guidance()
    elif args.play:
        play_with_ollama_and_rl(args.model)
    else:
        # Default: both train and play
        model, env = train_with_ollama_guidance()
        play_with_ollama_and_rl() 