# WHAT'S NEW

I added support for different LLMs to see how well they can play Minecraft. 

Other than GPT-4, the answer is not well.

Here are the LLMs I currently support:

| LLM                    | Embedding            | Performance                                                                                                                  | Notes                                                                                                                        |
|------------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| GPT-4 0125             | OpenAI Embeddings    | Supreme, acquired diamond pickaxe and beyond                                                                                 | -                                                                                                                            |
| GPT-4 Turbo 2024-04-09 | OpenAI Embeddings    | Excellent, acquired diamond pickaxe                                                                                          | It got a lot of 'failed to complete task: mine x wood' even though it succeeded in those tasks, not sure why                 |
| GPT-4 Turbo 0125       | OpenAI Embeddings    | Poor, makes iron tools at best                                                                                               | -                                                                                                                            |
| GPT-4o                 | OpenAI Embeddings    | Poor, makes tools pretty well but most of it's tasks are around exploration/finding stuff instead of advancing the tech tree |                                                                                                                              |
| GPT-4o Mini            | OpenAI Embeddings    | Poor, doesn't do much more than mining and smelting some ores.                                                               |                                                                                                                              |
| Gemini Pro 1.5         | models/embeddings001 | Poor, mines and smelts ores, but can't craft stuff well                                                                      | Using Google AI Studio/Google GenerativeAI                                                                                   |
| Gemini Pro 1.0         | models/embeddings001 | Very poor, doesn't go much farther than mining wood                                                                          | Using Google AI Studio/Google GenerativeAI                                                                                   |
| Mistral Large          | Mistral Embeddings   | Very poor, doesn't go much farther than mining wood                                                                          | -                                                                                                                            |
 | Claude 3.5 Sonnet | OpenAI Embeddings| Best in class, acquired diamonds by iteration 40 and continued to go from there, rarely failing to complete tasks | - |
| Claude 3 Opus          | OpenAI Embeddings    | Excellent, acquired diamond pickaxe, around the level of the new GPT-4 turbo API                                             | -                                                                                                                            |
| Claude 3 Sonnet        | OpenAI Embeddings    | Very poor, doesn't go much farther than mining wood                                                                          | Claude doesn't have an embedding model, so OpenAI is used. Using VertexAI model garden b/c I don't have an Anthropic API key |
| Command R+             | Cohere Embeddings    | Abysmal, couldn't even mine wood                                                                                             | Struggled with coding and JSON format                                                                                        |
| Vertex AI              | textgecko-003        | -                                                                                                                            | I have some support for the models on Vertex AI and the model garden, but it is still WIP                                    |

All of these LLMs were tested with the seed -2171441100564293352. GPT-4 was able to do pretty well in this seed, reliably mining diamonds and crafting a diamond pickaxe. 
For all intents and purposes, this is a good test seed, and if another model can get diamonds, it'd be comparable to GPT-4.


## Running

I couldn't get the Azure login to work, so I'm using the `mc_port` method. 

You can find the code I use to launch my experiments in `run.py`, just run the file (changing `mc_port` as needed).
Also, I keep all of my API keys in a `.env` file in the root directory (same as `run.py`). See this table for the LLMs and their 
corresponding environment variables and `model_name` (set in `run.py` for `action_agent_model_name`, `critic_agent_model_name`, etc.).

| LLM | Env Variable     | Model Name                     |
| --- |------------------|--------------------------------|
| GPT-4 0125 | `OPENAI_API_KEY` | `gpt-4-0125-preview`           |
| GPT-4 Turbo 2024-04-09 | `OPENAI_API_KEY` | `gpt-4-turbo-2024-04-09`       |
| GPT-4 Turbo 0125 | `OPENAI_API_KEY` | `gpt-4-turbo-preview`          |
| Gemini Pro 1.0 | `GOOGLE_API_KEY` | `models/gemini-1.0-pro-latest` |
| Mistral Large | `MISTRAL_API_KEY` | `mistral-large-latest`         |
| Claude Sonnet | `GOOGLE_APPLICATION_CREDENTIALS`[^1] | `claude-3-sonnet@20240229`     |
| Command R+ | `COHERE_API_KEY` | `command-r-plus`               |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS`[^1] | -                              |

[^1]: You need to set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to the path of your Application Default Credentials. See [here](https://cloud.google.com/docs/authentication/application-default-credentials) for more information.




# Voyager: An Open-Ended Embodied Agent with Large Language Models
<div align="center">

[[Website]](https://voyager.minedojo.org/)
[[Arxiv]](https://arxiv.org/abs/2305.16291)
[[PDF]](https://voyager.minedojo.org/assets/documents/voyager.pdf)
[[Tweet]](https://twitter.com/DrJimFan/status/1662115266933972993?s=20)

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://github.com/MineDojo/Voyager)
[![GitHub license](https://img.shields.io/github/license/MineDojo/Voyager)](https://github.com/MineDojo/Voyager/blob/main/LICENSE)
______________________________________________________________________


https://github.com/MineDojo/Voyager/assets/25460983/ce29f45b-43a5-4399-8fd8-5dd105fd64f2

![](images/pull.png)


</div>

We introduce Voyager, the first LLM-powered embodied lifelong learning agent
in Minecraft that continuously explores the world, acquires diverse skills, and
makes novel discoveries without human intervention. Voyager consists of three
key components: 1) an automatic curriculum that maximizes exploration, 2) an
ever-growing skill library of executable code for storing and retrieving complex
behaviors, and 3) a new iterative prompting mechanism that incorporates environment
feedback, execution errors, and self-verification for program improvement.
Voyager interacts with GPT-4 via blackbox queries, which bypasses the need for
model parameter fine-tuning. The skills developed by Voyager are temporally
extended, interpretable, and compositional, which compounds the agent’s abilities
rapidly and alleviates catastrophic forgetting. Empirically, Voyager shows
strong in-context lifelong learning capability and exhibits exceptional proficiency
in playing Minecraft. It obtains 3.3× more unique items, travels 2.3× longer
distances, and unlocks key tech tree milestones up to 15.3× faster than prior SOTA.
Voyager is able to utilize the learned skill library in a new Minecraft world to
solve novel tasks from scratch, while other techniques struggle to generalize.

In this repo, we provide Voyager code. This codebase is under [MIT License](LICENSE).

# Installation
Voyager requires Python ≥ 3.9 and Node.js ≥ 16.13.0. We have tested on Ubuntu 20.04, Windows 11, and macOS. You need to follow the instructions below to install Voyager.

## Python Install
```
git clone https://github.com/MineDojo/Voyager
cd Voyager
pip install -e .
```

## Node.js Install
In addition to the Python dependencies, you need to install the following Node.js packages:
```
cd voyager/env/mineflayer
npm install -g npx
npm install
cd mineflayer-collectblock
npx tsc
cd ..
npm install
```

## Minecraft Instance Install

Voyager depends on Minecraft game. You need to install Minecraft game and set up a Minecraft instance.

Follow the instructions in [Minecraft Login Tutorial](installation/minecraft_instance_install.md) to set up your Minecraft Instance.

## Fabric Mods Install

You need to install fabric mods to support all the features in Voyager. Remember to use the correct Fabric version of all the mods. 

Follow the instructions in [Fabric Mods Install](installation/fabric_mods_install.md) to install the mods.

# Getting Started
Voyager uses OpenAI's GPT-4 as the language model. You need to have an OpenAI API key to use Voyager. You can get one from [here](https://platform.openai.com/account/api-keys).

After the installation process, you can run Voyager by:
```python
from voyager import Voyager

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
azure_login = {
    "client_id": "YOUR_CLIENT_ID",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "[OPTIONAL] YOUR_SECRET_VALUE",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}
openai_api_key = "YOUR_API_KEY"

voyager = Voyager(
    azure_login=azure_login,
    openai_api_key=openai_api_key,
)

# start lifelong learning
voyager.learn()
```

* If you are running with `Azure Login` for the first time, it will ask you to follow the command line instruction to generate a config file.
* For `Azure Login`, you also need to select the world and open the world to LAN by yourself. After you run `voyager.learn()` the game will pop up soon, you need to:
  1. Select `Singleplayer` and press `Create New World`.
  2. Set Game Mode to `Creative` and Difficulty to `Peaceful`.
  3. After the world is created, press `Esc` key and press `Open to LAN`.
  4. Select `Allow cheats: ON` and press `Start LAN World`. You will see the bot join the world soon. 

# Resume from a checkpoint during learning

If you stop the learning process and want to resume from a checkpoint later, you can instantiate Voyager by:
```python
from voyager import Voyager

voyager = Voyager(
    azure_login=azure_login,
    openai_api_key=openai_api_key,
    ckpt_dir="YOUR_CKPT_DIR",
    resume=True,
)
```

# Run Voyager for a specific task with a learned skill library

If you want to run Voyager for a specific task with a learned skill library, you should first pass the skill library directory to Voyager:
```python
from voyager import Voyager

# First instantiate Voyager with skill_library_dir.
voyager = Voyager(
    azure_login=azure_login,
    openai_api_key=openai_api_key,
    skill_library_dir="./skill_library/trial1", # Load a learned skill library.
    ckpt_dir="YOUR_CKPT_DIR", # Feel free to use a new dir. Do not use the same dir as skill library because new events will still be recorded to ckpt_dir. 
    resume=False, # Do not resume from a skill library because this is not learning.
)
```
Then, you can run task decomposition. Notice: Occasionally, the task decomposition may not be logical. If you notice the printed sub-goals are flawed, you can rerun the decomposition.
```python
# Run task decomposition
task = "YOUR TASK" # e.g. "Craft a diamond pickaxe"
sub_goals = voyager.decompose_task(task=task)
```
Finally, you can run the sub-goals with the learned skill library:
```python
voyager.inference(sub_goals=sub_goals)
```

For all valid skill libraries, see [Learned Skill Libraries](skill_library/README.md).

# FAQ
If you have any questions, please check our [FAQ](FAQ.md) first before opening an issue.

# Paper and Citation

If you find our work useful, please consider citing us! 

```bibtex
@article{wang2023voyager,
  title   = {Voyager: An Open-Ended Embodied Agent with Large Language Models},
  author  = {Guanzhi Wang and Yuqi Xie and Yunfan Jiang and Ajay Mandlekar and Chaowei Xiao and Yuke Zhu and Linxi Fan and Anima Anandkumar},
  year    = {2023},
  journal = {arXiv preprint arXiv: Arxiv-2305.16291}
}
```

Disclaimer: This project is strictly for research purposes, and not an official product from NVIDIA.



## Improvements

This is where I could put a RAG system connected to the Minecraft Wiki:
- Skill library (looks like GPT-3.5 currently acts as environment feedback, but it would be helpful to hook up to a source of truth for other models)
- Critique: upon failing something, include info from the MC wiki on the context around it
- Writing code for task: once we give voyager a task like 'cract 1 stone pickaxe', we should probably include the MC wiki context to inform it

For multimodal processing, it would definitely be useful to send a screenshot (or 4, NSEW) of the current location to Gemini to help describe the situation
