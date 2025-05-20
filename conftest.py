import sys
import os

# Add the project root (the directory containing this conftest.py) to sys.path
# to ensure that modules like 'cli' and 'autogen' can be imported by tests.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
