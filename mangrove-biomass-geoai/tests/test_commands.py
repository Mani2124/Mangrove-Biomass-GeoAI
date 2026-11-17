import unittest
from pathlib import Path
from sensors import sentinel1, sentinel2, alos2
from gedi import labels
from ai import predict
from settings import ROOT

class Commands(unittest.TestCase):
    def test_engines_and_explicit_runs(self):
        run = Path('study with spaces')
        commands = [sentinel1.command(run,'2024-01-01','2024-02-01',2),
                    sentinel2.command(run,'2024-01-01','2024-02-01',2),
                    alos2.command(run,2024), predict.command(run,Path('model'))]
        for script, args in commands:
            self.assertTrue((ROOT/'engines'/script).exists())
            self.assertEqual(args[args.index('--run-dir')+1],run)
    def test_gedi_is_local(self):
        script, args = labels.command('local data','output')
        self.assertTrue((ROOT/'engines'/script).exists())
        self.assertEqual(args[1],'local data')

if __name__ == '__main__':
    unittest.main()
