import sys
import os

# 源码运行：入口在 src/；PyInstaller 打包：包在 sys._MEIPASS
if getattr(sys, "frozen", False):
    sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.tk_widget import run_app


def main():
    run_app()


if __name__ == '__main__':
    main()
