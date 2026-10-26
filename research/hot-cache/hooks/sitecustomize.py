# Research hooks for the hot-cache study. Mounted on PYTHONPATH by research/hot-cache/launch.sh.
# Chain the image's own sitecustomize first so nothing else changes.
import os

_system = '/usr/lib/python3.12/sitecustomize.py'
if os.path.exists(_system):
    with open(_system) as _f:
        exec(compile(_f.read(), _system, 'exec'))

if os.environ.get('QWEN38_RESEARCH') == '1':
    import qwen38_research
    qwen38_research.install()
