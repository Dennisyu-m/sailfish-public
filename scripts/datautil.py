"""数据处理公共函数：代码块抽取与语法检查、防污染索引、精确/近似去重指纹。"""
import ast
import hashlib
import json
import re
import zlib

import numpy as np

# ---------------------------------------------------------------- 代码检查
FENCE = re.compile(r"```([A-Za-z0-9_+#.-]*)[^\n]*\n(.*?)```", re.S)
PY_TAGS = {"", "python", "py", "python3", "Python", "Python3"}
OTHER_LANGS = {"java", "cpp", "c++", "c", "javascript", "js", "typescript", "ts", "go", "golang", "rust", "rs",
               "csharp", "c#", "cs", "php", "ruby", "rb", "swift", "kotlin", "scala", "r", "perl", "lua", "haskell",
               "julia", "dart", "objective-c", "objc", "vb", "matlab", "sql", "html", "css", "jsx", "tsx", "vue",
               "solidity", "elixir", "fortran", "zig", "nim", "groovy", "powershell", "verilog", "vhdl", "asm"}
MAX_CHARS = 60000


def parses(code: str) -> bool:
    if len(code) > MAX_CHARS:
        return False
    try:
        ast.parse(code)
        return True
    except Exception:  # noqa: BLE001  SyntaxError / RecursionError / MemoryError / ValueError
        return False


def markdown_python_ok(text: str):
    """markdown 文本：至少 1 个 Python 代码块且全部能 ast.parse；出现其他语言代码块则不要。
    返回 (ok, 代码块列表)。"""
    blocks = FENCE.findall(text)
    py = []
    for tag, code in blocks:
        t = tag.strip()
        if t.lower() in OTHER_LANGS:
            return False, []
        if t in PY_TAGS or t.lower() in {x.lower() for x in PY_TAGS}:
            if not parses(code):
                return False, []
            py.append(code)
    return len(py) > 0, py


# ---------------------------------------------------------------- 哈希工具
_P = np.uint64(1000003)
_word_cache: dict = {}


def word_ids(text: str) -> np.ndarray:
    ws = re.findall(r"\w+", text.lower())
    out = np.empty(len(ws), dtype=np.uint64)
    c = _word_cache
    for i, w in enumerate(ws):
        h = c.get(w)
        if h is None:
            h = zlib.crc32(w.encode()) + 1
            if len(c) < 2_000_000:
                c[w] = h
        out[i] = h
    return out


def ngram_hashes(w: np.ndarray, n: int) -> np.ndarray:
    if len(w) < n:
        return np.empty(0, dtype=np.uint64)
    m = len(w) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * _P + w[k:k + m]
    return h


def exact_hash(text: str) -> int:
    norm = re.sub(r"\s+", " ", text).strip().encode()
    return int.from_bytes(hashlib.blake2b(norm, digest_size=8).digest(), "little", signed=True)


# MinHash：64 个哈希函数，8 个带 x 8 行（相似度约 0.77 以上大概率碰撞）
N_PERM, N_BANDS = 64, 8
_rng = np.random.default_rng(12345)
_A = (_rng.integers(1, 2**63, N_PERM, dtype=np.uint64) | np.uint64(1))
_B = _rng.integers(0, 2**63, N_PERM, dtype=np.uint64)


def minhash_bands(w: np.ndarray) -> np.ndarray:
    sh = ngram_hashes(w, 5)
    if sh.size == 0:
        sh = ngram_hashes(w, max(1, len(w))) if len(w) else np.array([1], dtype=np.uint64)
    if sh.size > 4000:
        sh = np.unique(sh)
    with np.errstate(over="ignore"):
        sig = (sh[:, None] * _A[None, :] + _B[None, :]).min(axis=0)       # [64]
        sig = sig.reshape(N_BANDS, N_PERM // N_BANDS)
        bh = np.zeros(N_BANDS, dtype=np.uint64)
        for k in range(sig.shape[1]):
            bh = bh * _P + sig[:, k]
    return bh.view(np.int64)


# ---------------------------------------------------------------- 防污染
def split_args(s: str):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    names = []
    for a in out:
        a = a.split(":")[0].split("=")[0].strip().lstrip("*").strip()
        if a and a != "self" and a != "/":
            names.append(a)
    return tuple(names)


DEF_RE = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")


class Decontam:
    """§4.4：(1) 函数名 + 参数列表；(2) 13-gram（按词）重合。"""

    def __init__(self, he_path, mbpp_path):
        self.sigs = set()
        grams = []
        for path, kind in ((he_path, "he"), (mbpp_path, "mbpp")):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                src = r["prompt"] + r["canonical_solution"] if kind == "he" else r["canonical_solution"]
                for name, args in DEF_RE.findall(src):
                    if name == r["entry_point"]:
                        self.sigs.add((name, split_args(args)))
                text = r["prompt"] + "\n" + r["canonical_solution"]
                grams.append(ngram_hashes(word_ids(text), 13))
        self.grams = np.unique(np.concatenate(grams))
        self.names = {n for n, _ in self.sigs}

    def name_hit(self, text: str) -> bool:
        if "def" not in text:
            return False
        for name, args in DEF_RE.findall(text):
            if name in self.names and (name, split_args(args)) in self.sigs:
                return True
        return False

    def ngram_hit(self, w: np.ndarray) -> bool:
        g = ngram_hashes(w, 13)
        if g.size == 0:
            return False
        idx = np.searchsorted(self.grams, g)
        idx[idx >= self.grams.size] = 0
        return bool((self.grams[idx] == g).any())
