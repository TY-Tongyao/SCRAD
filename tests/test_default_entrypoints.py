import json
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]

class DefaultEntrypointTests(unittest.TestCase):

    def test_top_level_training_executes_paper_settings_for_five_seeds(self):
        for dataset, lr, corpus in [('questions', 0.01, 100000), ('reddit', 0.001, 200000)]:
            result = subprocess.run([sys.executable, str(ROOT / 'train.py'), '--dataset', dataset, '--output', str(ROOT / 'runs/dry_run_regression'), '--dry-run'], capture_output=True, text=True, check=True)
            commands = [shlex.split(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(commands), 21)
            training_seeds = []
            for command in commands:
                options = {command[i]: command[i + 1] for i in range(len(command) - 1) if command[i].startswith('--') and (not command[i + 1].startswith('--'))}
                self.assertEqual(options['--protocol'], 'paper')
                if 'scrad_complete.pretrain' in command:
                    self.assertEqual(options['--graph-mode'], 'full')
                    self.assertEqual(options['--hidden'], '256')
                    self.assertEqual(options['--epochs'], '0')
                    self.assertEqual(float(options['--lr']), lr)
                if 'scrad_complete.prepare' in command:
                    self.assertEqual(options['--graph-mode'], 'full')
                    self.assertEqual(options['--codebook-size'], '512')
                    self.assertEqual(int(options['--corpus-size']), corpus)
                if 'scrad_complete.train' in command:
                    self.assertEqual(options['--epochs'], '50')
                    self.assertEqual(options['--max-steps'], '0')
                    self.assertEqual(options['--train-views'], '9')
                    self.assertEqual(float(options['--encoder-lr']), lr)
                    self.assertEqual(float(options['--token-lr']), lr)
                    self.assertEqual(float(options['--vq-weight']), 0.0)
                    self.assertEqual(Path(options['--llm']).name, 'DeepSeek-R1-Distill-Qwen-14B')
                    training_seeds.append(int(options['--seed']))
            self.assertEqual(training_seeds, [42, 43, 44, 45, 46])

    def test_default_configuration_has_no_legacy_training_stages(self):
        for dataset in ('questions', 'reddit'):
            config = json.loads((ROOT / 'configs' / f'{dataset}.json').read_text())
            self.assertNotIn('single300', config)
            self.assertNotIn('joint25', config)
            self.assertEqual(config['infer']['protocol'], 'paper')
            self.assertEqual(config['compatibility_infer']['protocol'], 'compatibility')

    def test_awq_inference_requires_explicit_compatibility_mode(self):
        for extra, protocol, model in [([], 'paper', 'DeepSeek-R1-Distill-Qwen-14B'), (['--protocol', 'compatibility'], 'compatibility', 'DeepSeek-R1-Distill-Qwen-14B-AWQ')]:
            result = subprocess.run([sys.executable, str(ROOT / 'infer.py'), '--dataset', 'questions', '--dry-run', *extra], capture_output=True, text=True, check=True)
            argv = json.loads(result.stdout)['argv']
            self.assertEqual(argv[argv.index('--protocol') + 1], protocol)
            self.assertEqual(Path(argv[argv.index('--llm') + 1]).name, model)
if __name__ == '__main__':
    unittest.main()
