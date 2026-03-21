# ISAA Reinforcement Learning System

## Übersicht

Das RL-System (`isaa_mod/base/rl/`) ermöglicht Agent-Training durch Reinforcement Learning.

## Komponenten

| Komponente | Datei | Beschreibung |
|------------|-------|---------------|
| `Training` | `training.py` | Haupttraining Loop |
| `DataCollection` | `data_collection.py` | Experience Sammlung |
| `DatasetBuilder` | `dataset_builder.py` | Trainings-Dataset |
| `RewardFunctions` | `reward_functions.py` | Reward-Berechnung |
| `AgentTools` | `agent_tools.py` | Tools für RL-Agents |

## Training Flow

```
1. Agent führt Task aus
2. Environment gibt Reward
3. Experience wird gesammelt
4. Dataset wird aufgebaut
5. Policy wird aktualisiert
```

## Usage

```python
from isaa_mod.base.rl import Training, RewardFunctions

# Reward Function definieren
def my_reward(context):
    if context.success:
        return 1.0
    elif context.partial:
        return 0.5
    else:
        return -0.1

# Training starten
training = Training(
    agent=agent,
    reward_function=my_reward
)

await training.run(episodes=100)
```

## Hardware Support

- **GPU Training** - via hardware_config.py
- **Batch Processing** - Mehrere Agents parallel
- **Distributed** - Multi-GPU Support
