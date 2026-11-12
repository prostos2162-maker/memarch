"""Стара назва модуля «обруч» (hoop). Тепер — uroboros.py («Уроборос»). Лишено як аліас для старого коду."""
from .uroboros import *  # noqa: F401,F403
from .uroboros import init, predict, select, loss, stats, next_budget, n_slots  # noqa: F401
