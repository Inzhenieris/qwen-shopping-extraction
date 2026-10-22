import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import benchmark as b

ROW = {'id': 1, 'sentence': 'Buy 3 apples.', 'gold': [{'item': 'apples', 'quantity': 3}],
       'scenario': {'kind': 'requests', 'private': 'NEVER_SEND_THIS'}}

class Tests(unittest.TestCase):
    def test_strict_schema(self):
        for raw in ['{}', '{"items":[{"name":"a","quantity":true}]}',
                    '{"items":[{"name":"a","quantity":0}]}',
                    '{"items":[{"name":"a","quantity":1.0}]}',
                    '{"items":[{"name":"a","quantity":1},{"name":"a","quantity":2}]}',
                    '{"items":[],"extra":1}', '{"items":[],"items":[]}']:
            self.assertFalse(b.parse_prediction(raw)[1])
        for raw in ['text {"items":[]}', '```json\n{"items":[]}\n```']:
            self.assertFalse(b.parse_prediction(raw)[0])
        self.assertTrue(b.parse_prediction('{"items":[]}')[1])

    def test_order_and_no_normalization(self):
        row = {**ROW, 'gold': [{'item': 'apples', 'quantity': 3}, {'item': 'pears', 'quantity': 2}]}
        with patch.object(b, 'infer', return_value=('{"items":[{"name":"pears","quantity":2},{"name":"apples","quantity":3}]}', 'stop')):
            self.assertTrue(b.evaluate(row, 'model', 1)['exact_match'])
        with patch.object(b, 'infer', return_value=('{"items":[{"name":"apple","quantity":3}]}', 'stop')):
            r=b.evaluate(ROW, 'model', 1)
            self.assertEqual((r['tp'],r['fp'],r['fn']), (0,1,1))

    def test_quantity_and_errors_denominator(self):
        with patch.object(b, 'infer', return_value=('{"items":[{"name":"apples","quantity":2}]}', 'stop')):
            wrong=b.evaluate(ROW,'m',1)
        with patch.object(b,'infer',side_effect=TimeoutError()):
            failed=b.evaluate(ROW,'m',1)
        m=b.metrics([wrong,failed])
        self.assertEqual(m['examples'],2)
        self.assertEqual(m['transport_or_envelope_errors'],1)
        self.assertEqual(m['fn'],2)
        self.assertEqual(m['exact_match_accuracy'],0)

    def test_sentence_only_no_gold_leak(self):
        with patch.object(b,'api',return_value={'done':True,'message':{'content':'{"items":[]}'}}) as api:
            b.evaluate(ROW,'m',1)
            payload=api.call_args.args[1]
            self.assertEqual(payload['messages'],[{'role':'system','content':b.PROMPT},{'role':'user','content':ROW['sentence']}])
            self.assertNotIn('NEVER_SEND_THIS',json.dumps(payload))
            self.assertNotIn('gold',json.dumps(payload))
            self.assertNotIn('format',payload)

    def test_snapshot_and_run_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.jsonl';p.write_text((json.dumps(ROW)+'\n')*100)
            self.assertEqual(len(b.snapshot(p)[1]),100)
            with patch.object(b,'api',return_value={'models':[{'name':'m'}]}), patch.object(b,'infer',return_value=('{"items":[{"name":"apples","quantity":3}]}','stop')):
                args=['--dataset',str(p),'--model','m','--limit','3','--run-dir',str(Path(tmp)/'run')]
                self.assertEqual(b.main(args),0)
                self.assertEqual(b.main(args),1)
                summary=json.loads((Path(tmp)/'run/summary.json').read_text())
                self.assertEqual(summary['exact_matches'],3)
                self.assertTrue(summary['source_unchanged_after_run'])
            p.write_text(json.dumps(ROW)+'\n')
            with self.assertRaises(ValueError): b.snapshot(p)

if __name__=='__main__': unittest.main()
