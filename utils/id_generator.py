from time import time
from typing import Union, Optional, Any


class RealtimeId(str):
    """
    A specialized ID class that encodes time information and supports comparison.
    
    The ID format is: {prefix}_{timestamp}
    
    This class inherits from str for backward compatibility but adds
    comparison functionality based on timestamp.
    """
    
    def __new__(cls, prefix: str, timestamp: Optional[float] = None):
        """Create a new RealtimeId instance."""
        timestamp = timestamp or time()
        # Create the actual string representation
        id_str = f'{prefix}_{timestamp}'
        # Create the instance from the string
        instance = super().__new__(cls, id_str)
        # Store attributes
        instance._prefix = prefix
        instance._timestamp = timestamp
        return instance
    
    @classmethod
    def from_string(cls, id_string: str) -> 'RealtimeId':
        """Create a RealtimeId from an existing ID string."""
        try:
            prefix, timestamp_str = id_string.split('_', 1)
            timestamp = float(timestamp_str)
            return cls(prefix, timestamp)
        except (ValueError, IndexError):
            # If the format is not as expected, return a new ID with the whole 
            # string as prefix and current time
            return cls(id_string)
    
    @property
    def prefix(self) -> str:
        """Get the prefix part of the ID."""
        return self._prefix
    
    @property
    def timestamp(self) -> float:
        """Get the timestamp part of the ID."""
        return self._timestamp
    
    def __lt__(self, other):
        if isinstance(other, RealtimeId):
            return self._timestamp < other._timestamp
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, RealtimeId):
            return self._timestamp <= other._timestamp
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, RealtimeId):
            return self._timestamp > other._timestamp
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, RealtimeId):
            return self._timestamp >= other._timestamp
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, RealtimeId):
            return self._timestamp == other._timestamp and self._prefix == other._prefix
        if isinstance(other, str):
            # For backward compatibility, compare with string
            return str(self) == other
        return NotImplemented


def generateId(prefix: str = 'id', timestamp: Optional[float] = None) -> RealtimeId:
    """
    Generate a timestamped ID that can be compared
    :param prefix: Prefix for the ID
    :param timestamp: Optional timestamp to use (defaults to current time)
    :return: RealtimeId instance (compatible with str)
    """
    return RealtimeId(prefix, timestamp)


def generateId(prefix: str = 'id', timestamp: Optional[float] = None) -> RealtimeId:
    """
    Generate a timestamped ID that can be compared
    :param prefix: Prefix for the ID
    :param timestamp: Optional timestamp to use (defaults to current time)
    :return: RealtimeId instance (compatible with str)
    """
    return RealtimeId(prefix, timestamp)
    
