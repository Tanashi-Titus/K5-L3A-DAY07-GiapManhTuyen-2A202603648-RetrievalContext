"""Offline checks for Gemini tool-call and response normalization."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from gemini_controller import GeminiController


class Response:
    def __init__(self, body): self.body = json.dumps(body).encode('utf-8')
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, *args): return self.body


class GeminiControllerTests(unittest.TestCase):
    @patch('gemini_controller.urlopen')
    def test_forces_first_search_and_parses_tool_call(self, urlopen):
        urlopen.return_value = Response({'candidates':[{'content':{'parts':[{'functionCall':{'name':'search','args':{'query':'physics'}}}]}}]})
        controller = GeminiController('local-test-key', max_searches=1)
        self.assertEqual(controller.decide('question', []), {'action':'search','query':'physics'})
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload['toolConfig']['functionCallingConfig']['mode'], 'ANY')

    @patch('gemini_controller.urlopen')
    def test_disables_search_at_limit_and_extracts_citations(self, urlopen):
        urlopen.return_value = Response({'candidates':[{'content':{'parts':[{'text':'Answer [chunk-1]'}]}}]})
        controller = GeminiController('local-test-key', max_searches=1)
        result = controller.decide('question', [{'context':'evidence', 'queries':['first']}])
        self.assertEqual(result['citations'], ['chunk-1'])
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload['toolConfig']['functionCallingConfig']['mode'], 'NONE')

    @patch('gemini_controller.urlopen')
    def test_separates_grouped_citations_only_when_ids_are_in_context(self, urlopen):
        urlopen.return_value = Response({'candidates':[{'content':{'parts':[{'text':'Answer [a, b]'}]}}]})
        controller = GeminiController('local-test-key', max_searches=1)
        result = controller.decide('question', [{'context':'evidence', 'queries':['first'], 'context_ids':['a','b']}])
        self.assertEqual(result['answer'], 'Answer [a] [b]')
        self.assertEqual(result['citations'], ['a','b'])
        self.assertEqual(result['raw_answer'], 'Answer [a, b]')

    @patch('gemini_controller.urlopen')
    def test_missing_tool_query_uses_original_question_and_records_reason(self, urlopen):
        urlopen.return_value = Response({'candidates':[{'content':{'parts':[{'functionCall':{'name':'search','args':{}}}]}}]})
        controller = GeminiController('local-test-key')
        result = controller.decide('original question', [])
        self.assertEqual(result['query'], 'original question')
        self.assertIn('omitted query', result['fallback_reason'])

    @patch('gemini_controller.urlopen')
    def test_group_with_unknown_id_is_not_silently_accepted(self, urlopen):
        urlopen.return_value = Response({'candidates':[{'content':{'parts':[{'text':'Answer [a, unknown]'}]}}]})
        controller = GeminiController('local-test-key')
        result = controller.decide('question', [{'context':'evidence', 'queries':['q'], 'context_ids':['a']}])
        self.assertEqual(result['citations'], ['a, unknown'])


if __name__ == '__main__': unittest.main()
