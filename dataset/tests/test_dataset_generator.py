import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import urllib.error
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dataset_generator as d


class GeneratorTests(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(len(d.ITEMS), 100)
        self.assertEqual(len(set(d.ITEMS)), 100)

    def test_seed_and_random_default(self):
        def batch(seed):
            rng = random.Random(seed)
            return [d.make_scenario(rng) for _ in range(20)]
        self.assertEqual(batch(42), batch(42))
        self.assertNotEqual(batch(42), batch(43))
        self.assertNotEqual(batch(None), batch(None))

    def test_bounds_and_kinds(self):
        rng = random.Random(1)
        quantities = set()
        for kind in d.KINDS:
            for _ in range(100):
                s = d.make_scenario(rng, 3, 9, kind)
                names = [r['item'] for r in s['requests']]
                self.assertTrue(set(names + s['exclusions']) <= set(d.ITEMS))
                for r in s['requests']:
                    self.assertIs(type(r['quantity']), int)
                    self.assertTrue(3 <= r['quantity'] <= 9)
                    quantities.add(r['quantity'])
                self.assertEqual(len(names) > len(set(names)), kind in ('duplicates', 'mixed'))
                self.assertEqual(bool(s['exclusions']), kind in ('exclusions', 'mixed'))
        self.assertEqual(quantities, set(range(3, 10)))
        with self.assertRaises(ValueError):
            d.make_scenario(rng, 0, 10)

    def test_aggregation_exclusion_and_empty(self):
        s = {'requests': [{'item': 'apples', 'quantity': 70},
                          {'item': 'apples', 'quantity': 80},
                          {'item': 'pears', 'quantity': 5}], 'exclusions': ['pears', 'bananas']}
        self.assertEqual(d.gold_for(s), [{'item': 'apples', 'quantity': 150}])
        s['exclusions'].append('apples')
        self.assertEqual(d.gold_for(s), [])

    def test_dry_run_without_network_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, 'request_json') as net:
            path = Path(tmp) / 'data.jsonl'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(d.main(['--dry-run', '--count', '100', '--seed', '17', '--output', str(path)]), 0)
                original = path.read_bytes()
                self.assertEqual(d.main(['--dry-run', '--output', str(path)]), 1)
            self.assertEqual(path.read_bytes(), original)
            rows = [json.loads(x) for x in original.splitlines()]
            self.assertEqual(len(rows), 100)
            self.assertEqual(set(r['scenario']['kind'] for r in rows), set(d.KINDS))
            for r in rows:
                self.assertEqual(r['gold'], d.gold_for(r['scenario']))
                self.assertIsNone(r['sentence'])
            net.assert_not_called()

    def test_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(d.os.environ, {}, clear=True), patch.object(d, 'load_env', return_value={}), patch.object(d, 'request_json') as net:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(d.main(['--output', str(Path(tmp) / 'data')]), 1)
            net.assert_not_called()
            self.assertFalse((Path(tmp) / "data").exists())

    def test_model_must_match(self):
        with patch.object(d, 'request_json', return_value={'data': [{'id': d.MODEL, 'name': 'Flash'}]}):
            self.assertEqual(d.verify_model(d.MODEL)['id'], d.MODEL)
            with self.assertRaises(d.GenerationError):
                d.verify_model('nonexistent')
        with patch.object(d, 'request_json', return_value=[]):
            with self.assertRaises(d.GenerationError):
                d.verify_model(d.MODEL)

    def test_render_shape_and_no_gold_in_prompt(self):
        scenario = d.make_scenario(random.Random(2), kind='mixed')
        def result(content, reason='stop'):
            return {'choices': [{'finish_reason': reason, 'message': {'content': content}}]}
        with patch.object(d, 'request_json', return_value=result(json.dumps({'sentence': 'Buy things.'}))) as api:
            self.assertEqual(d.render_sentence(scenario, d.MODEL, 'fake-test-key'), 'Buy things.')
            sent = json.loads(api.call_args.args[1]['messages'][1]['content'])
            self.assertEqual(sent, scenario)
            self.assertNotIn('gold', sent)
        for bad in [None, {}, {'choices': []}, result('not json'), result('{}'),
                    result('[]'), result('{"sentence": 4}'), result('{"sentence":" "}'),
                    result('{"sentence":"ok","gold":[]}'), result('{"sentence":"ok"}', 'length')]:
            with self.subTest(bad=bad), patch.object(d, 'request_json', return_value=bad):
                with self.assertRaises(d.GenerationError):
                    d.render_sentence(scenario, d.MODEL, 'fake-test-key')

    def test_http_retry_after(self):
        err = urllib.error.HTTPError('https://example.test', 429, 'rate limit', {'Retry-After': '7'}, None)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        with patch.object(d.urllib.request, 'urlopen', side_effect=[err, response]) as call, patch.object(d.time, 'sleep') as sleep:
            self.assertEqual(d.request_json('https://example.test', retries=1), {'ok': True})
            self.assertEqual(call.call_count, 2)
            sleep.assert_called_once_with(7)
        with patch.object(d.time, 'time', return_value=0):
            self.assertEqual(d.retry_delay('Thu, 01 Jan 1970 00:00:10 GMT', 0), 10)
        self.assertEqual(d.retry_delay('NaN', 0), 1)
        self.assertEqual(d.retry_delay('inf', 0), 1)
        with self.assertRaises(d.GenerationError):
            d.retry_delay('121', 0)

    def test_finite_retries_and_auth_failure(self):
        for error, expected in [(urllib.error.URLError('offline'), 3),
                                (TimeoutError(), 3),
                                (urllib.error.HTTPError('x', 401, 'secret response', {}, None), 1)]:
            with patch.object(d.urllib.request, 'urlopen', side_effect=error) as call, patch.object(d.time, 'sleep'):
                with self.assertRaises(d.GenerationError) as caught:
                    d.request_json('https://example.test', retries=2)
                self.assertEqual(call.call_count, expected)
                self.assertNotIn('secret response', str(caught.exception))

    def test_400_diagnostics_redact_credentials_and_echoes(self):
        secret = 'my-private-key-value'
        prompt = 'Private shopping instruction with quantities'
        body = {'error': {'code': 400, 'message': 'Invalid response_format ' + secret + ' ' + prompt,
                         'metadata': {'provider_name': 'Example', 'raw': json.dumps({'error': {
                             'message': 'Unsupported parameter sk-or-v1-fakecredential Bearer hiddenvalue',
                             'param': 'response_format', 'Authorization': secret, 'messages': [prompt]}})},
                         'headers': {'Authorization': secret}, 'messages': [prompt]}}
        err = urllib.error.HTTPError('https://example.test', 400, 'bad', {}, io.BytesIO(json.dumps(body).encode()))
        with patch.object(d.urllib.request, 'urlopen', side_effect=err) as call:
            with self.assertRaises(d.GenerationError) as caught:
                d.request_json('https://example.test', {'messages': [{'content': prompt}]}, secret, retries=0)
            detail = str(caught.exception)
            self.assertIn('400', detail)
            self.assertIn('response_format', detail)
            self.assertIn('Example', detail)
            for forbidden in (secret, prompt, 'sk-or-v1-fakecredential', 'hiddenvalue', 'Authorization'):
                self.assertNotIn(forbidden, detail)
            call.assert_called_once()

    def test_diagnostics_bounded_and_unstructured_omitted(self):
        for raw in (b'<html>private</html>', b'x' * 17000,
                    json.dumps({'error': {'message': 'Invalid input messages: private fragment'}}).encode()):
            err = urllib.error.HTTPError('x', 400, 'bad', {}, io.BytesIO(raw))
            detail = d.http_error_detail(err)
            self.assertLessEqual(len(detail), 1801)
            self.assertNotIn('private', detail)

    def test_env_loading_and_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = root / '.env'
            env.write_text("# comment\nexport OPENROUTER_API_KEY='file-token'\nOPENROUTER_MODEL=file/model # note\nIGNORED=$(false)\n")
            with patch.object(d, 'PROJECT_DIR', root), patch.dict(d.os.environ, {}, clear=True):
                self.assertEqual(d.settings(), ('file-token', 'file/model'))
                with patch.dict(d.os.environ, {'OPENROUTER_API_KEY': 'process-token', 'OPENROUTER_MODEL': 'process/model'}):
                    self.assertEqual(d.settings(), ('process-token', 'process/model'))
                    self.assertEqual(d.settings('cli/model'), ('process-token', 'cli/model'))
                    self.assertEqual(d.settings(env_file=env), ('file-token', 'file/model'))
                    self.assertEqual(d.settings('cli/model', env), ('file-token', 'cli/model'))
            env.write_text('OPENROUTER_API_KEY="unterminated-secret')
            with self.assertRaises(d.GenerationError) as caught:
                d.load_env(env)
            self.assertNotIn('unterminated-secret', str(caught.exception))
            env.write_text('OPENROUTER_API_KEY=" invalid token "')
            with self.assertRaises(d.GenerationError):
                d.settings(env_file=env)

    def test_first_failure_no_output_and_later_failure_partial(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, 'settings', return_value=('test', d.MODEL)), patch.object(d, 'verify_model', return_value={'id': d.MODEL}):
            path = Path(tmp) / 'data.jsonl'
            with patch.object(d, 'render_sentence', side_effect=d.GenerationError('failure')), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(d.main(['--count', '2', '--output', str(path)]), 1)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with patch.object(d, 'render_sentence', side_effect=['Buy apples.', d.GenerationError('failure')]), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(d.main(['--count', '2', '--output', str(path)]), 1)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_non_json_error_classification_no_echo(self):
        raw = b'<html>Cloudflare Bad Request Authorization: secretvalue</html>'
        err = urllib.error.HTTPError('x', 400, 'bad', {}, io.BytesIO(raw))
        detail = d.http_error_detail(err)
        self.assertIn('non-JSON', detail)
        self.assertIn('edge/cloudflare', detail)
        self.assertNotIn('secretvalue', detail)
        self.assertNotIn('Authorization', detail)

    def test_progress_counts_and_separate_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'data.jsonl'
            err, stdout = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(stdout):
                self.assertEqual(d.main(['--dry-run', '--count', '2', '--output', str(output)]), 0)
            log = err.getvalue()
            self.assertIn('Starting:', log)
            self.assertIn('Model verification: skipped', log)
            self.assertIn('Example 1/2: scenario generation starting', log)
            self.assertIn('Saved 1/2', log)
            self.assertIn('Saved 2/2', log)
            self.assertIn('ETA', log)
            self.assertIn('Success: saved 2/2', log)
            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(len([json.loads(x) for x in output.read_text().splitlines()]), 2)
            self.assertNotIn('Starting:', output.read_text())

    def test_quiet_and_failure_actual_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'quiet.jsonl'
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(d.main(['--quiet', '--dry-run', '--count', '1', '--output', str(output)]), 0)
            self.assertEqual(err.getvalue(), '')
            output = Path(tmp) / 'partial.jsonl'
            with patch.object(d, 'settings', return_value=('secret-test-credential', d.MODEL)), patch.object(d, 'verify_model', return_value={'id': d.MODEL}), patch.object(d, 'render_sentence', side_effect=['private sentence', d.GenerationError('failure')]), contextlib.redirect_stderr(err):
                self.assertEqual(d.main(['--quiet', '--count', '3', '--output', str(output)]), 1)
            self.assertIn('Saved 1/3', err.getvalue())
            self.assertNotIn('secret-test-credential', err.getvalue())
            self.assertNotIn('private sentence', err.getvalue())
            self.assertNotIn('Starting:', err.getvalue())
            self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_retry_logging_and_flush(self):
        for quiet in (False, True):
            token = d.QUIET.set(quiet)
            try:
                err = urllib.error.HTTPError('x', 429, 'sensitive body', {'Retry-After': '2'}, None)
                response = MagicMock()
                response.__enter__.return_value.read.return_value = b'{}'
                log = io.StringIO()
                with patch.object(d.urllib.request, 'urlopen', side_effect=[err, response]), patch.object(d.time, 'sleep'), contextlib.redirect_stderr(log):
                    d.request_json('https://example.test', key='secret-test-credential', retries=1)
                if quiet:
                    self.assertEqual(log.getvalue(), '')
                else:
                    self.assertEqual(log.getvalue(), 'Retry: HTTP 429; attempt 2/2 in 2.0s\n')
                self.assertNotIn('secret-test-credential', log.getvalue())
                self.assertNotIn('sensitive body', log.getvalue())
            finally:
                d.QUIET.reset(token)
        with patch('builtins.print') as printer:
            d.progress('test')
            self.assertTrue(printer.call_args.kwargs['flush'])

    def test_bad_json(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'not json'
        with patch.object(d.urllib.request, 'urlopen', return_value=response):
            with self.assertRaises(d.GenerationError):
                d.request_json('https://example.test')

    def test_invalid_cli(self):
        for args in [['--count', '0'], ['--min-quantity', '10', '--max-quantity', '1'],
                     ['--retries', '-1'], ['--timeout', '0'], ['--max-tokens', '1']]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    d.main(['--output', '/unused'] + args)


if __name__ == '__main__':
    unittest.main()
