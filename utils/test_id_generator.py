import time
import unittest
from utils.id_generator import generateId, RealtimeId


class TestRealtimeId(unittest.TestCase):
    def test_id_creation(self):
        """Test basic ID creation"""
        id1 = generateId("test")
        self.assertTrue(isinstance(id1, RealtimeId))
        self.assertTrue(str(id1).startswith("test_"))
        self.assertEqual(id1.prefix, "test")
        self.assertIsInstance(id1.timestamp, float)
        
    def test_from_string(self):
        """Test creating ID from string"""
        # Create from valid format
        str_id = "test_1620000000.0"
        id_obj = RealtimeId.from_string(str_id)
        self.assertEqual(id_obj.prefix, "test")
        self.assertEqual(id_obj.timestamp, 1620000000.0)
        
        # Create from invalid format
        invalid_str = "invalid-format"
        id_obj = RealtimeId.from_string(invalid_str)
        self.assertEqual(id_obj.prefix, "invalid-format")
        self.assertNotEqual(id_obj.timestamp, 0)
    
    def test_string_compatibility(self):
        """Test string compatibility"""
        id1 = generateId("test")
        # Should be usable as a string
        self.assertTrue(isinstance(id1, str))
        # String operations should work
        self.assertTrue(id1.startswith("test_"))
        
    def test_comparison(self):
        """Test comparison operations"""
        # Create IDs with controlled timestamps
        id1 = RealtimeId("test", 1000.0)
        id2 = RealtimeId("test", 2000.0)
        id3 = RealtimeId("test", 1000.0)  # Same as id1
        
        # Test comparison operators
        self.assertTrue(id1 < id2)
        self.assertTrue(id2 > id1)
        self.assertTrue(id1 <= id3)
        self.assertTrue(id1 >= id3)
        self.assertTrue(id1 == id3)
        self.assertFalse(id1 == id2)
        
        # Different prefixes but same timestamp
        id4 = RealtimeId("other", 1000.0)
        self.assertFalse(id1 == id4)
        
        # String comparison (backward compatibility)
        self.assertTrue(id1 == str(id1))
    
    def test_time_ordering(self):
        """Test that IDs are properly time-ordered"""
        # Create multiple IDs in sequence
        ids = []
        for _ in range(3):
            ids.append(generateId("test"))
            time.sleep(0.01)  # Ensure different timestamps
        
        # Check they're ordered by creation time
        self.assertTrue(ids[0] < ids[1] < ids[2])
        
        # Sort and verify order
        shuffled = [ids[1], ids[2], ids[0]]
        sorted_ids = sorted(shuffled)
        self.assertEqual(sorted_ids, [ids[0], ids[1], ids[2]])


if __name__ == "__main__":
    unittest.main()
