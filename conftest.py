import sys
import os

# Ensure the repo root is on sys.path so 'api' and 'models' are importable
# regardless of where pytest is invoked from (local, CI, Docker).
sys.path.insert(0, os.path.dirname(__file__))
