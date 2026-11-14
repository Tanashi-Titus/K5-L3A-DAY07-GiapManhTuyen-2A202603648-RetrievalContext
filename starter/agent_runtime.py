"""Run a whole agent case on one provider; switch providers on rate limits."""
import os
import time
from urllib.error import URLError
from agentic import run_agent
from gemini_controller import GeminiController
from groq_controller import GroqController
from openai_controller import OpenAIController


class AgentRunError(RuntimeError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


class RecordingRetriever:
    def __init__(self, retriever): self.retriever, self.calls = retriever, []
    def search(self, query, *, filters, top_k):
        results = self.retriever.search(query, filters=filters, top_k=top_k)
        self.calls.append({'event':'search','query':query,'filter':filters,'results':results})
        return results


def available_providers():
    providers={}
    if os.getenv('OPENAI_API_KEY'):
        providers['openai']=(OpenAIController,os.environ['OPENAI_API_KEY'],os.getenv('OPENAI_MODEL') or 'gpt-5.4-mini')
    if os.getenv('GEMINI_API_KEY'):
        providers['gemini']=(GeminiController,os.environ['GEMINI_API_KEY'],os.getenv('GEMINI_MODEL') or 'gemini-3.5-flash-lite')
    if os.getenv('GROQ_API_KEY'):
        providers['groq']=(GroqController,os.environ['GROQ_API_KEY'],os.getenv('GROQ_MODEL') or 'openai/gpt-oss-120b')
    return providers


def run_with_fallback(question, retriever, *, provider='openai', top_k=3, budget=1600, max_searches=2):
    providers=available_providers()
    if provider not in providers: raise ValueError(f'Missing API key for {provider}')
    fallback_order=() if provider=='openai' else ('gemini','groq','openai')
    order=[provider]+[name for name in fallback_order if name in providers and name!=provider]
    attempts=[]
    for index in range(max(2,2*len(order))):
        name=order[index % len(order)]
        cls,key,model=providers[name]
        controller=cls(key,model,max_searches,max_retries=0)
        recorder=RecordingRetriever(retriever)
        try:
            result=run_agent(question,controller,recorder,top_k=top_k,budget=budget,max_searches=max_searches)
            return result,controller,name,attempts
        except Exception as exc:
            message=f'{type(exc).__name__}: {exc}'
            attempts.append({'provider':name,'model':model,'error':message,'usage':controller.usage,
                             'controller_decisions':controller.decisions,'events':recorder.calls})
            if ('HTTP 429' not in message and 'HTTP 503' not in message
                    and not isinstance(exc, (TimeoutError, ConnectionError, URLError))):
                raise AgentRunError(message,attempts) from exc
            if index % len(order) == len(order)-1 and index+1 < max(2,2*len(order)):
                time.sleep(30)
    raise AgentRunError('All available providers are rate limited or temporarily unavailable',attempts)
