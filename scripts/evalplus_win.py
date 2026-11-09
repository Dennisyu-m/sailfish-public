"""
在 Windows 上跑 evalplus.evaluate：它用到 Unix 才有的 resource 模块和 SIGALRM。
- resource：换成空实现（内存上限在 Windows 上不生效），evalplus 其余的安全限制照常。
- time_limit（SIGALRM）：换成空实现；单题超时由 evalplus 外层进程的 join/kill 兜底。
这段补丁放在模块顶层，spawn 出来的子进程重新导入本脚本时也会生效。
用法同 python -m evalplus.evaluate。
"""
import contextlib
import sys
import types

if sys.platform == "win32":
    fake = types.ModuleType("resource")
    fake.RLIMIT_AS, fake.RLIMIT_DATA, fake.RLIMIT_STACK = 0, 1, 2
    fake.setrlimit = lambda *a, **k: None
    sys.modules["resource"] = fake

    import evalplus.eval as _ev
    import evalplus.eval.utils as _ut

    @contextlib.contextmanager
    def _no_time_limit(seconds):
        yield

    _ut.time_limit = _no_time_limit
    _ev.time_limit = _no_time_limit

if __name__ == "__main__":
    from evalplus.evaluate import main
    main()
