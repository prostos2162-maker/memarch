"""Стара назва тесту обруча. Тепер tests/test_uroboros.py; цей файл лише запускає його (аліас)."""
import os
import runpy

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_uroboros.py"), run_name="__main__")
