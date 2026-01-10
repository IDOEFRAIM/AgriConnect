import unittest
import os
import shutil
import sys
import numpy as np

sys.path.append(os.getcwd())

from rag.components.vector_store import VectorStoreHandler

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for tests
        self.test_dir = "tests/temp_data"
        self.index_path = f"{self.test_dir}/test.index"
        self.meta_path = f"{self.test_dir}/test_meta.json"
        
        self.handler = VectorStoreHandler(
            index_path=self.index_path,
            metadata_path=self.meta_path,
            dimension=4 # Small dimension for testing
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_documents(self):
        # Create dummy documents
        doc1 = {"text": "Test 1", "source_type": "TEST", "vector": [1.0, 0.0, 0.0, 0.0]}
        doc2 = {"text": "Test 2", "source_type": "TEST", "vector": [0.0, 1.0, 0.0, 0.0]}
        
        self.handler.add_documents([doc1, doc2])
        
        # Verify count
        self.assertEqual(self.handler.index.ntotal, 2)
        # Verify metadata
        self.assertEqual(len(self.handler.metadata), 2)
        self.assertEqual(self.handler.metadata[0]["text"], "Test 1")

    def test_delete_by_source(self):
        doc1 = {"text": "Keep Me", "source_type": "KEPT", "vector": [1.0, 0.0, 0.0, 0.0]}
        doc2 = {"text": "Delete Me", "source_type": "TEMP", "vector": [0.0, 1.0, 0.0, 0.0]}
        
        self.handler.add_documents([doc1, doc2])
        self.assertEqual(self.handler.index.ntotal, 2)
        
        self.handler.delete_by_source("TEMP")
        
        self.assertEqual(self.handler.index.ntotal, 1)
        # Check that the remaining doc is "KEPT"
        remaining = list(self.handler.metadata.values())[0]
        self.assertEqual(remaining["source_type"], "KEPT")

if __name__ == '__main__':
    unittest.main()
