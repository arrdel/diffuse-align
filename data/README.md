# Data Directory

## Required Datasets

DiffuseAlign uses multi-agent trajectory data collected by running LLM agents in task environments.

### Step 1: Collect Seed Trajectories

Run the trajectory collection script to generate training data using GPT-4o agents:

```bash
python scripts/generate_trajectories.py \
    --env alfworld \
    --num_episodes 5000 \
    --num_agents 2 \
    --output data/alfworld_multi/
```

### Step 2: Data Format

Each trajectory is stored as a JSON file:

```json
{
  "task_id": "alf_001",
  "task_description": "Find the red mug and place it on the kitchen counter.",
  "num_agents": 2,
  "agent_roles": ["navigator", "manipulator"],
  "steps": [
    {
      "step": 0,
      "agent_id": 0,
      "action": "navigate",
      "args": "kitchen",
      "utterance": "I'll head to the kitchen to look around.",
      "observation": "You arrive in the kitchen."
    }
  ],
  "success": true,
  "metadata": {"optimal_steps": 8, "difficulty": "moderate"}
}
```

### Benchmarks

1. **ALFWorld-Multi** — Household tasks (2 agents)
2. **WebArena-Multi** — Web navigation tasks (3 agents)  
3. **CollabCooking** — Collaborative cooking (2 agents)

Note: WebArena and ALFWorld require their respective environment installations.
See the main README for setup instructions.
