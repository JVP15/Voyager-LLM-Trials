import os

import anthropic._types
from anthropic import AnthropicVertex
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, messages_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import HarmBlockThreshold, HarmCategory

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings, VertexAIModelGarden

# thansk to https://gist.github.com/gregburek/1441055 for rate limiting code
import time
import reka

def RateLimited(maxPerSecond):
    minInterval = 1.0 / float(maxPerSecond)
    def decorate(func):
        lastTimeCalled = [0.0]
        def rateLimitedFunction(*args,**kargs):
            elapsed = time.time() - lastTimeCalled[0]
            leftToWait = minInterval - elapsed
            if leftToWait>0:
                time.sleep(leftToWait)
            ret = func(*args,**kargs)
            lastTimeCalled[0] = time.time()
            return ret
        return rateLimitedFunction
    return decorate



def convert_chat_message_to_anthropic(message): # had to look at the code for Langchain's ChatAnthropic and work backwards
    role = message.type
    content = message.content

    if role == 'human':
        role = 'user'
    elif role == 'ai':
        role = 'assistant'

    return {'role': role, 'content': content}

def convert_chat_messages_to_anthropic(messages):
    system = anthropic._types.NOT_GIVEN

    anthropic_messages = []

    for message in messages:
        if message.type == 'system':
            system = message.content
        else:
            anthropic_messages.append(
                convert_chat_message_to_anthropic(message)
            )

    return anthropic_messages, system

class ChatVertexAIAnthropic(BaseChatModel):
    model_name: str = None
    temperature: float = 0.0
    request_timeout: int = 60

    client: AnthropicVertex = None

    def __init__(self, model_name, temperature=anthropic._types.NOT_GIVEN, timeout=anthropic._types.NOT_GIVEN):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = timeout

        if 'opus' or '3.5' in model_name:
            region = 'us-east5'
        else:
            region = 'us-central1'

        self.client = AnthropicVertex(region=region, project_id=os.environ['VERTEX_AI_PROJECT'])

    @RateLimited(5) # right now, I can only call this a few times/second
    def _generate_rate_limited(self, messages, stop=None, run_manager=None, **kwargs):
        messages, system = convert_chat_messages_to_anthropic(messages)

        response = self.client.messages.create(
            max_tokens=2048,
            system=system,
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            timeout=self.request_timeout
        )

        response = response.content[0].text

        response = AIMessage(response)

        return ChatResult(generations=[ChatGeneration(message=response)])

    def _generate(self,
                  messages,
                  stop = None,
                  run_manager = None,
                  **kwargs
                  ):

        # I may still get an error here, so re-try it up to 3 times, and if that doesn't work, then just raise the error
        for i in range(3):
            try:
                return self._generate_rate_limited(messages, stop, run_manager, **kwargs)
            except Exception as e:
                if i < 2:
                    time.sleep(30) # wait a little bit before trying again if it fails the second time, hopefully this will give it a chance to recover fully
                if i == 2:
                    raise e

                print(f'Got error {e}, trying {2 - i} more times')


    @property
    def _llm_type(self) -> str:
        return "vertexai-anthropic"

def convert_chat_message_to_reka(message): # had to look at the code for Langchain's ChatAnthropic and work backwards
    role = message.type
    content = message.content
    if role == 'system':
        role = 'human'
    elif role == 'ai':
        role = 'model'

    return {'type': role, 'text': content}

def convert_chat_messages_to_reka(messages):
    reka_messages = []

    for i in range(len(messages)):
        if i == 0 and messages[i].type == 'system' and messages[i+1].type == 'human':
            # Combine system and human messages
            combined_content = messages[i].content + "\n\n" + messages[i+1].content
            reka_messages.append({'type': 'human', 'text': combined_content})
        elif messages[i-1].type == 'system' and messages[i].type == 'human':
            # Skip this message as it has been combined with the previous system message
            continue
        else:
            reka_messages.append(convert_chat_message_to_reka(messages[i]))

    return reka_messages


class ChatReka(BaseChatModel):
    model_name: str = None
    temperature: float = 0.0
    request_timeout: int = 60
    api_key: str = None

    def __init__(self, model_name, temperature=None, timeout=None):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = timeout
        reka.API_KEY = os.environ['REKA_API_KEY']

    def _generate_rate_limited(self, messages, stop=None, run_manager=None, **kwargs):
        messages = convert_chat_messages_to_reka(messages)

        response = reka.chat(
            model_name=self.model_name,
            conversation_history=messages,
            temperature=self.temperature,
        )
        response = response['text']

        response = AIMessage(response)

        return ChatResult(generations=[ChatGeneration(message=response)])

    def _generate(self,
                  messages,
                  stop = None,
                  run_manager = None,
                  **kwargs
                  ):
        return self._generate_rate_limited(messages, stop, run_manager, **kwargs)

        # # I may still get an error here, so re-try it up to 3 times, and if that doesn't work, then just raise the error
        # for i in range(3):
        #     try:
        #         return self._generate_rate_limited(messages, stop, run_manager, **kwargs)
        #     except Exception as e:
        #         if i < 2:
        #             time.sleep(30) # wait a little bit before trying again if it fails the second time, hopefully this will give it a chance to recover fully
        #         if i == 2:
        #             raise e
        #
        #         print(f'Got error {e}, trying {2 - i} more times')

    @property
    def _llm_type(self) -> str:
        return "chat-reka"



# class ChatSaveInputOutputLLM(BaseChatModel):
#     def __init__(self, llm):
#         super().__init__()
#         self.llm = llm
#         self.inputs = []
#         self.outputs = []
#
#     def _generate(self, messages, stop=None, run_manager=None, **kwargs):
#         self.inputs.append(messages)
#         result = self.llm._generate(messages, stop, run_manager, **kwargs)
#         self.outputs.append(result)
#         return result
#
#     @property
#     def _llm_type(self):
#         return self.llm._llm_type

def get_llm(model_name, temperature=None, request_timeout=None):
    if 'gpt' in model_name:
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    elif 'gemini' in model_name:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    elif 'command' in model_name:
        llm = ChatCohere(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    elif '<vertexai>' in model_name:
        raise NotImplementedError("VertexAI models are not supported in this function (it's just super buggy and annoying to use Claude, perhaps I'll use Gemini when it is better).")
        # see this issue when it comes to using Claude models on vertex ai https://github.com/langchain-ai/langchain/discussions/19442
        # model_name = model_name.replace('<vertexai>', '')
        # llm = ChatVertexAI(
        #     model_name=model_name,
        #     temperature=temperature,
        #     convert_system_message_to_human=True,
        # )
    elif 'mistral' in model_name:
        llm = ChatMistralAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    elif 'claude' in model_name:
        llm = ChatVertexAIAnthropic(
            model_name=model_name,
            temperature=temperature,
            timeout=request_timeout
        )
    elif 'reka' in model_name:
        llm = ChatReka(
            model_name=model_name,
            temperature=temperature,
            timeout=request_timeout
        )
    else:
        raise(NotImplementedError(f"Model {model_name} not implemented"))

    #llm = ChatSaveInputOutputLLM(llm)

    return llm


def get_embedding_model(model_name):

    if 'gemini' in model_name:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001'
        )
        #embedding_model = OpenAIEmbeddings()
    elif 'gpt' in model_name:
        embedding_model = OpenAIEmbeddings()
    elif '<vertexai>' in model_name:
        embedding_model = VertexAIEmbeddings('textembedding-gecko@003')
    elif 'mistral' in model_name:
        embedding_model = MistralAIEmbeddings()
    elif 'claude' in model_name:
        embedding_model = OpenAIEmbeddings() # claude doen't have an embedding model, so I guess I'll just use OpenAI
    elif 'command' in model_name:
        embedding_model = CohereEmbeddings()
    elif 'reka' in model_name:
        embedding_model = OpenAIEmbeddings() # reka doesn't have an embedding model, so I guess I'll just use OpenAI
    else:
        raise(NotImplementedError(f"Model {model_name} not implemented for embeddings"))

    return embedding_model

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    from langchain_core.prompts import ChatPromptTemplate

    #llm = ChatVertexAIAnthropic(model_name='claude-3-sonnet@20240229', )
    llm = ChatReka(model_name='reka-core')
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are an expert at python programming'),
        ('human', '{text}')
    ])
    chain = prompt | llm
    out = chain.invoke({'text': 'Explain the GIL in python'})
    print(type(out), out)


    #
    # # even this doesn't work for some reason, maybe Langchain just doesn't have full support for model gardens
    # llm = VertexAIModelGarden(project=os.environ['VERTEX_AI_PROJECT'], location='us-central1-a', endpoint_id='publishers/anthropic/models/claude-3-sonnet@20240229')
    #
    # print(llm.invoke('Explain the GIL in python and how to work around it'))

