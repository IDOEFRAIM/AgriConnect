import unittest
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

def run_suite():
    loader = unittest.TestLoader()
    # Discover tests in 'tests/tools' and 'tests/rag' 
    # Excluding 'tests/test_orchestrator.py' to focus on bricks first
    suite_tools = loader.discover('tests/tools', pattern='test_*.py')
    suite_rag = loader.discover('tests/rag', pattern='test_*.py')
    
    suite = unittest.TestSuite([suite_tools, suite_rag])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Running Bricks Tests (Tools + RAG)...")
    run_suite()
