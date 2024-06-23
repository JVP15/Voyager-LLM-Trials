from voyager import Voyager
import dotenv
import os

import dotenv
dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

voyager = Voyager(
    #azure_login=azure_login,
    mc_port=49224,
    openai_api_key=openai_api_key,
    ckpt_dir='skill_library/claude-3-5-sonnet-2171441100564293352',
    #resume=True,
    #ckpt_dir='skill_library/claude--2171441100564293352',
    # action_agent_model_name='gpt-4-0125-preview',
    # critic_agent_model_name='gpt-4-0125-preview',
    # curriculum_agent_model_name='gpt-4-0125-preview',
    # curriculum_agent_qa_model_name='gpt-4-0125-preview',
    # skill_manager_model_name='gpt-4-0125-preview',

    # action_agent_model_name='gpt-4-turbo',
    # critic_agent_model_name='gpt-4-turbo',
    # curriculum_agent_model_name='gpt-4-turbo',
    # curriculum_agent_qa_model_name='gpt-4-turbo',
    # skill_manager_model_name='gpt-4-turbo',

    # action_agent_model_name='gpt-4o',
    # critic_agent_model_name='gpt-4o',
    # curriculum_agent_model_name='gpt-4o',
    # curriculum_agent_qa_model_name='gpt-4o',
    # skill_manager_model_name='gpt-4o',

    # action_agent_model_name='models/gemini-1.0-pro-latest',
    # critic_agent_model_name='models/gemini-1.0-pro-latest',
    # curriculum_agent_model_name='models/gemini-1.0-pro-latest',
    # curriculum_agent_qa_model_name='models/gemini-1.0-pro-latest',
    # skill_manager_model_name='models/gemini-1.0-pro-latest',

    # action_agent_model_name='models/gemini-1.5-flash',
    # critic_agent_model_name='models/gemini-1.5-flash',
    # curriculum_agent_model_name='models/gemini-1.5-flash',
    # curriculum_agent_qa_model_name='models/gemini-1.5-flash',
    # skill_manager_model_name='models/gemini-1.5-flash',

    # action_agent_model_name='mistral-large-latest',
    # critic_agent_model_name='mistral-large-latest',
    # curriculum_agent_model_name='mistral-large-latest',
    # curriculum_agent_qa_model_name='mistral-large-latest',
    # skill_manager_model_name='mistral-large-latest',

    # action_agent_model_name='claude-3-sonnet@20240229',
    # critic_agent_model_name='claude-3-sonnet@20240229',
    # curriculum_agent_model_name='claude-3-sonnet@20240229',
    # curriculum_agent_qa_model_name='claude-3-sonnet@20240229',
    # skill_manager_model_name='claude-3-sonnet@20240229',

    action_agent_model_name='claude-3-5-sonnet@20240620',
    critic_agent_model_name='claude-3-5-sonnet@20240620',
    curriculum_agent_model_name='claude-3-5-sonnet@20240620',
    curriculum_agent_qa_model_name='claude-3-5-sonnet@20240620',
    skill_manager_model_name='claude-3-5-sonnet@20240620',

    # action_agent_model_name='claude-3-opus@20240229',
    # critic_agent_model_name='claude-3-opus@20240229',
    # curriculum_agent_model_name='claude-3-opus@20240229',
    # curriculum_agent_qa_model_name='claude-3-opus@20240229',
    # skill_manager_model_name='claude-3-opus@20240229',


    # action_agent_model_name='command-r-plus',
    # critic_agent_model_name='command-r-plus',
    # curriculum_agent_model_name='command-r-plus',
    # curriculum_agent_qa_model_name='command-r-plus',
    # skill_manager_model_name='command-r-plus',

    # action_agent_model_name='reka-core',
    # critic_agent_model_name='reka-core',
    # curriculum_agent_model_name='reka-core',
    # curriculum_agent_qa_model_name='reka-core',
    # skill_manager_model_name='reka-core',
)

# start lifelong learning
voyager.learn()


# seed for training: -2171441100564293352 (Gpt-4 turbo didn't do well on this seed)
# for gemini pro, use 'models/gemini-1.0-pro-latest' for the latest model
# looks like I can use gemini ultra w/ 'models/gemini-1.0-ultra-latest' and gemini pro 1.5 w/ 'gemini-1.5-pro-latest', best to use 'models/embedding' for embedding models,
# gpt-4 should be 'gpt-4-0125-preview', gpt-4-turbo (worse) should be 'gpt-4-turbo-preview'
# for mistral, it is 'mistral-large-latest'
# for claude, it's 'claude-3-sonnet@20240229'
