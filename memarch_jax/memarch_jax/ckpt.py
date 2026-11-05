"""M11: чекпойнт стану тренування — params, opt (стан Instinct), tau, step + data_state (JSON від data.Corpus) +
довільні meta. Два бекенди: "orbax" (StandardCheckpointer, якщо orbax встановлено) і "npz" (по файлу на листок,
bf16 зберігається як uint16-бітовий вигляд → побітовий round-trip). "auto" = orbax, якщо імпортується.

Тека ckpt:
  meta.json   — {"backend", "step", "leaves": [{"path": [[тип, ключ], ...], "dtype", "shape"}], "data_state", "extra"}
  arrays/     — npz: <номер листка>.npy; orbax: каталог StandardCheckpointer
Запис атомарний: у <dir>.tmp, далі rename (spot-машину можуть забрати посеред запису).

torch↔jax: import_torch(sd) → (params, frozen, tau) через convert.torch_to_jax; export_torch(params, frozen, tau)
→ {torch-ім'я: numpy fp32} через convert.jax_to_torch (повний не-SHARD ckpt, W=1)."""
import json
import os
import shutil

import numpy as np
import jax
import jax.numpy as jnp

from . import convert

_ML = {"bfloat16": jnp.bfloat16}


def _orbax():
    try:
        import orbax.checkpoint as ocp
        return ocp
    except Exception:
        return None


# ---------------------------------------------------------------- структура дерева ↔ JSON

def _key(k):
    if hasattr(k, "key"):
        return ["d", k.key]
    if hasattr(k, "idx"):
        return ["l", k.idx]
    raise TypeError(f"непідтримуваний ключ {k!r}")


def _flatten(tree):
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return [[_key(k) for k in p] for p, _ in leaves], [x for _, x in leaves]


def _empties(tree, path=()):
    """Шляхи порожніх dict/list (jax.tree_util їх не бачить як листки, а Instinct чекає, напр., cs={})."""
    out = []
    if isinstance(tree, dict):
        if not tree:
            return [[list(path), "d"]]
        for k, v in tree.items():
            out += _empties(v, path + (["d", k],))
    elif isinstance(tree, (list, tuple)):
        if not tree:
            return [[list(path), "l"]]
        for i, v in enumerate(tree):
            out += _empties(v, path + (["l", i],))
    return out


def _unflatten(paths, vals, empties=()):
    """Будуємо вкладені dict/list за шляхами (списки — за індексами)."""
    root = {"__list__": False}
    items = list(zip(paths, vals)) + [(p, {"__list__": t == "l"}) for p, t in empties]
    for path, v in items:
        if not path:
            return v
        node = root
        for j, (t, k) in enumerate(path):
            k = k if t == "d" else int(k)
            if j == len(path) - 1:
                node[k] = v
            else:
                node = node.setdefault(k, {"__list__": path[j + 1][0] == "l"})
    return _lists(root)


def _lists(n):
    if not isinstance(n, dict) or "__list__" not in n:
        return n
    is_list = n.pop("__list__")
    n = {k: _lists(v) for k, v in n.items()}
    return [n[i] for i in range(len(n))] if is_list else n


def _np_dtype(name):
    return _ML.get(name, np.dtype(name) if name in np.sctypeDict else name)


# ---------------------------------------------------------------- save / restore

def state_tree(state):
    """Масивна частина стану (без python-об'єктів)."""
    return {k: state[k] for k in ("params", "opt", "tau", "step") if k in state}


def save(path, state, data_state=None, extra=None, backend="auto"):
    """state — dict з params/opt/tau/step (як step.init_state). data_state — JSON-серіалізований стан корпусу."""
    ocp = _orbax()
    if backend == "auto":
        backend = "orbax" if ocp is not None else "npz"
    tree = jax.device_get(state_tree(state))
    paths, vals = _flatten(tree)
    tmp = path.rstrip("/\\") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    meta = dict(backend=backend, empties=_empties(tree), step=int(np.asarray(tree.get("step", 0))), data_state=data_state, extra=extra,
                leaves=[dict(path=p, dtype=str(np.asarray(v).dtype), shape=list(np.shape(v)))
                        for p, v in zip(paths, vals)])
    if backend == "npz":
        os.makedirs(os.path.join(tmp, "arrays"))
        for i, v in enumerate(vals):
            a = np.asarray(v)
            if a.dtype == jnp.bfloat16:
                a = a.view(np.uint16)                         # біти bf16 як є
            np.save(os.path.join(tmp, "arrays", f"{i}.npy"), a)
    elif backend == "orbax":
        ck = ocp.StandardCheckpointer()
        ck.save(os.path.abspath(os.path.join(tmp, "arrays")), {str(i): jnp.asarray(v) for i, v in enumerate(vals)})
        ck.wait_until_finished()
    else:
        raise ValueError(backend)
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    return path


def restore(path):
    """→ (state {params, opt, tau, step} як jnp-масиви, data_state, extra)."""
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    lv = meta["leaves"]
    if meta["backend"] == "npz":
        vals = []
        for i, d in enumerate(lv):
            a = np.load(os.path.join(path, "arrays", f"{i}.npy"))
            if d["dtype"] == "bfloat16":
                a = a.view(jnp.bfloat16)
            vals.append(jnp.asarray(a))
    else:
        ocp = _orbax()
        assert ocp is not None, "ckpt записано orbax, а orbax не встановлено"
        like = {str(i): jax.ShapeDtypeStruct(tuple(d["shape"]), _np_dtype(d["dtype"])) for i, d in enumerate(lv)}
        got = ocp.StandardCheckpointer().restore(os.path.abspath(os.path.join(path, "arrays")), like)
        vals = [jnp.asarray(got[str(i)]) for i in range(len(lv))]
    for v, d in zip(vals, lv):
        assert str(v.dtype) == d["dtype"] and list(v.shape) == d["shape"], (d, v.dtype, v.shape)
    state = _unflatten([d["path"] for d in lv], vals, meta.get("empties", []))
    return state, meta["data_state"], meta["extra"]


def latest(root):
    """Найновіший ckpt у теці root з підтеками step_<N> (None — немає)."""
    if not os.path.isdir(root):
        return None
    steps = [int(n[5:]) for n in os.listdir(root)
             if n.startswith("step_") and n[5:].isdigit() and os.path.exists(os.path.join(root, n, "meta.json"))]
    return os.path.join(root, f"step_{max(steps)}") if steps else None


# ---------------------------------------------------------------- torch ↔ jax

def import_torch(sd, dtype=jnp.bfloat16):
    """torch state_dict (FullLM2, повний) → (params у dtype, frozen {lock_w, ind_a}, tau | None)."""
    return convert.torch_to_jax(sd, dtype)


def export_torch(params, frozen=None, tau=None):
    """params (jax) → {torch-ім'я: numpy fp32} (для torch.save / load_state_dict(strict=False))."""
    return convert.jax_to_torch(params, frozen, tau)
