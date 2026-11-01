# Скрипт суперника для G4: оригінал nanogpt_medium_g4.py + оцінка нашим протоколом (OUR_VAL), більше нічого.
src = open("nanogpt_medium_g4.py", encoding="utf-8").read()
def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a[:60], src.count(a)); src = src.replace(a, b)
exec(open("make_pc_ourval.py", encoding="utf-8").read())
open("nanogpt_g4_ourval.py", "w", encoding="utf-8").write(src)
print("ok")
