"""Read-only snapshot: never runs the queue or loads a model."""
from complex_common import *


def main():
    state = read_json(RUN / 'progress.json')
    print(f'RUN: {RUN}\nStatus: {state["status"]}\nStage: {state.get("stage", "")}', flush=True)
    for method in METHODS:
        print(f'{method}: {state.get("actual_updates", {}).get(method, 0)}/600 actual updates')
    print(f'Inference: {state.get("completed_trajectories", 0)}/40; current={state.get("current_job", "none")}')
    print(f'Updated: {state["updated_at"]}\nResults: {RUN / "00_看结果"}')
    if state.get('pid'):
        try:
            os.kill(state['pid'], 0)
            print(f'Queue process {state["pid"]}: alive')
        except ProcessLookupError:
            print(f'Queue process {state["pid"]}: not alive; snapshot may be stale')


if __name__ == '__main__':
    main()
