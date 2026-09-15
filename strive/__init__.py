"""STRIVE streaming architecture prototype."""
import os
import sys

__version__ = "0.2.0"


def _repair_shadowed_site_packages() -> None:
    """Ensure this virtualenv's site-packages precedes loader-injected paths.

    Datadog's APM host injection installs a system-wide LD_PRELOAD shim via
    /etc/ld.so.preload, which prepends its own bundled site-packages to
    sys.path for every Python process on the machine - ahead of the active
    virtualenv, and regardless of PYTHONPATH or the DD_* environment
    variables. Its vendored typing_extensions predates `sentinel`, so
    `import fastapi` dies inside anyio with:

        ImportError: cannot import name 'sentinel' from 'typing_extensions'

    We reorder sys.path so our pinned dependencies win, and drop any module
    already imported from one of those foreign paths so the corrected order
    is actually used. Injected paths stay importable, just last.
    """
    if sys.prefix == sys.base_prefix:
        return  # not running inside a virtualenv; nothing to protect
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(sys.prefix, "lib", version, "site-packages")
    if site_packages not in sys.path:
        return
    ours = sys.path.index(site_packages)
    # Anything ahead of us belonging to neither the venv, the base
    # interpreter (stdlib), nor this repository was injected from outside.
    # The repo root must be excluded explicitly: sys.prefix is the .venv
    # directory, so the project directory that contains it does not match.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    safe = (sys.base_prefix, sys.prefix, repo_root)
    foreign = [
        path for path in sys.path[:ours]
        if path and not os.path.abspath(path).startswith(safe)
    ]
    if not foreign:
        return
    for path in foreign:
        sys.path.remove(path)
        sys.path.append(path)
    # A foreign copy already imported would still win from sys.modules, so
    # evict those and let them re-import from the corrected path order.
    # ddtrace itself is left alone: it is mid-initialisation under the
    # preload shim and evicting it breaks the tracer.
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] == "ddtrace":
            continue
        origin = getattr(module, "__file__", None)
        if origin and origin.startswith(tuple(foreign)):
            del sys.modules[name]


_repair_shadowed_site_packages()
