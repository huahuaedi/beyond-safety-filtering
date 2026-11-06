import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from summarize_posttrain import METRICS, read_metrics


class PosttrainTest(unittest.TestCase):
    def test_selection_rejects_safety_regression_and_incomplete_eval(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'val_tokens.json').write_text('["a", "b"]')
            for label,score,nc in [('baseline',.8,1),('50',.9,1),('100',.99,.5),('150',.85,1),('200',.88,1)]:
                name=root.name+'_val_'+label; path=root/'metrics'/name/'scores.csv'; path.parent.mkdir(parents=True)
                with path.open('w') as f:
                    w=csv.DictWriter(f,fieldnames=['token','valid',*METRICS]); w.writeheader()
                    for token in ['a','b']:
                        row=dict.fromkeys(METRICS,1.); row.update(score=score,no_at_fault_collisions=nc,token=token,valid=True); w.writerow(row)
            proc=subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_posttrain.py')),'--run',str(root),'--stage','select','--baseline','/base'],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(json.loads((root/'selected.json').read_text())['selected'],'50')
            with self.assertRaises(ValueError): read_metrics(root,root.name+'_val_baseline',['a','b','missing'])

    def test_navsim_process_failure_is_not_zero_reward(self):
        source=Path(__file__).with_name('rl_stage1_prototype.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='score_with_navsim_simulator')
        scope=dict(NAVSIM_SCORE_CALLS=0,NAVSIM_SCORED_TRAJECTORIES=0,np=np,NAVSIM_ENV_PYTHON='/python',NAVSIM_METRIC_CACHE_PATH='/cache',NAVSIM_WORKER_SCRIPT='/worker',SUB_SCORE_KEYS=['score'],execution_trace_args=lambda *args: [])
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<score-function>','exec'),scope)
        with patch('subprocess.run',return_value=subprocess.CompletedProcess([],1,stderr='broken environment')):
            with self.assertRaisesRegex(RuntimeError,'broken environment'):
                scope['score_with_navsim_simulator']('scene',np.zeros((2,8,3)))


if __name__=='__main__':unittest.main()
