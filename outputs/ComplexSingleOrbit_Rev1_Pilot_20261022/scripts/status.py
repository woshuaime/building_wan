"""Read-only one-line status; never initializes CUDA or launches work."""
from complex_common import *

p=read_json(RUN/'progress.json')
print(now(),p['status'],f'{p["completed_trajectories"]}/{p["expected_trajectories"]} trajectories',
      p.get('current_job'), 'updates',p['actual_updates'])
path=RUN/'logs'/f'{p.get("current_job")}.log'
if path.exists():
    lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
    for line in lines[-3:]:print(line)
