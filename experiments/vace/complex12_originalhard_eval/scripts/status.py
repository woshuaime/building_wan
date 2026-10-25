"""Read-only progress; never starts CUDA or schedules inference."""
from complex_common import *

if __name__=='__main__':
    print(json.dumps(read_json(RUN/'progress.json'),ensure_ascii=False,indent=2))
    print(RUN/'00_看结果')
