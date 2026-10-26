"""Read-only status; never launches a worker."""
from complex_common import *

if __name__ == '__main__':
    state = read_json(RUN / 'progress.json')
    print(json.dumps(state, ensure_ascii=False, indent=2))
    print(RUN / '00_看结果/README_从这里开始.md')
