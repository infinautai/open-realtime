"""
Test script to validate RealtimeId Pydantic integration.

This script tests that:
1. RealtimeId can be used in Pydantic models
2. RealtimeId serialization/deserialization works correctly
3. Pydantic schema generation for RealtimeId works correctly
"""
import sys
from time import time
from typing import Union, Optional, List
import pydantic
from utils.id_generator import RealtimeId, generateId
from pydantic import BaseModel, Field
import json

print(f"Pydantic version: {pydantic.__version__}")

# Test model using RealtimeId
class TestModel(BaseModel):
    id: Union[str, RealtimeId] = Field(default_factory=lambda: generateId("test"))
    name: str
    timestamp: float = Field(default_factory=time)
    items: List[Union[str, RealtimeId]] = []
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

def test_realtimeid_integration():
    # Test 1: Create a model with RealtimeId
    model = TestModel(name="Test Model")
    print(f"Model created with ID: {model.id}")
    
    # Test 2: Verify RealtimeId serialization
    model_json = model.model_dump_json()
    print(f"Serialized model: {model_json}")
    
    # Test 3: Verify RealtimeId deserialization
    serialized = json.loads(model_json)
    
    # Create a new model from the serialized data
    deserialized = TestModel.model_validate(serialized)
    print(f"Deserialized model: {deserialized.model_dump_json()}")
    
    # Test 4: Verify that the IDs match
    assert str(deserialized.id) == str(model.id), f"Deserialized ID {deserialized.id} does not match original {model.id}"
    
    # Test 5: Add items with both string and RealtimeId
    model.items.append(generateId("item1"))
    model.items.append("plain_string_id")
    model.items.append(generateId("item2"))
    
    # Test 6: Serialize and deserialize again with the items
    model_json = model.model_dump_json()
    print(f"Model with items: {model_json}")
    
    deserialized = TestModel.model_validate(json.loads(model_json))
    print(f"Deserialized model with items: {deserialized.model_dump_json()}")
    
    # Test 7: Verify items were preserved
    assert len(deserialized.items) == len(model.items), "Item count mismatch"
    
    # Test 8: Try to generate JSON schema
    try:
        schema = TestModel.model_json_schema()
        print(f"JSON Schema: {json.dumps(schema, indent=2)}")
    except Exception as e:
        print(f"Schema generation error: {e}")
        raise
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_realtimeid_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        sys.exit(1)
