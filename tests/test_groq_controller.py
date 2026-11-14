"""Offline tests of Groq tool calling and provider failover."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'starter'))
from agent_runtime import run_with_fallback
from groq_controller import GroqController


class Response:
    def __init__(self, body): self.body=json.dumps(body).encode()
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,*args): return self.body


class GroqTests(unittest.TestCase):
    @patch('groq_controller.urlopen')
    def test_tool_call_and_auth_header(self, urlopen):
        urlopen.return_value=Response({'choices':[{'message':{'tool_calls':[{'function':{'name':'search','arguments':'{"query":"physics"}'}}]}}]})
        c=GroqController('test-key',max_searches=1)
        self.assertEqual(c.decide('question',[])['query'],'physics')
        request=urlopen.call_args.args[0]
        self.assertEqual(request.get_header('User-agent'),'groq-python/0.18.0')
        self.assertEqual(json.loads(request.data)['tool_choice'],'required')

    @patch('groq_controller.urlopen')
    def test_unicode_citations_are_normalized_only_if_in_context(self, urlopen):
        urlopen.return_value=Response({'choices':[{'message':{'content':'Answer 【a】【b】'}}]})
        c=GroqController('test-key',max_searches=1)
        result=c.decide('question',[{'queries':['q'],'context':'source','context_ids':['a','b']}])
        self.assertEqual(result['answer'],'Answer [a][b]')
        self.assertEqual(result['citations'],['a','b'])
        payload=json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn('tools',payload)
        self.assertNotIn('tool_choice',payload)

    @patch('agent_runtime.run_agent')
    @patch('agent_runtime.available_providers')
    def test_rate_limit_restarts_case_with_other_provider(self, providers, run_agent):
        class Controller:
            def __init__(self,key,model,max_searches,max_retries=0):
                self.model=model; self.usage=[]; self.decisions=[]
        providers.return_value={'gemini':(Controller,'a','gemini-test'),'groq':(Controller,'b','groq-test')}
        run_agent.side_effect=[RuntimeError('HTTP 429'),{'status':'answered','events':[],'citations':[]}]
        result,controller,provider,attempts=run_with_fallback('q',object(),provider='gemini')
        self.assertEqual((result['status'],controller.model,provider),('answered','groq-test','groq'))
        self.assertEqual(attempts[0]['provider'],'gemini')
        self.assertEqual(run_agent.call_count,2)


if __name__ == '__main__': unittest.main()
