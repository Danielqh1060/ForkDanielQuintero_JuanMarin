import hashlib
import numpy as np # Importar numpy aquí también

class BloomFilter:
    def __init__(self, capacity: int, error_rate: float):
        if not (0 < error_rate < 1):
            raise ValueError("Error rate must be between 0 and 1.")
        if not capacity > 0:
            raise ValueError("Capacity must be > 0.")

        self.capacity = capacity
        # Calculate optimal number of bits (m) and hash functions (k)
        self.num_bits = int(-(capacity * np.log(error_rate)) / (np.log(2) ** 2))
        self.num_hash_functions = int((self.num_bits / capacity) * np.log(2))
        
        # Ensure at least one hash function
        if self.num_hash_functions == 0:
            self.num_hash_functions = 1

        self.bit_array = np.zeros(self.num_bits, dtype=bool)

    def _hash(self, item, seed):
        # Using built-in hash plus a seed for multiple hash functions
        # For tuples/sets, ensure they are hashable (e.g., convert sets to frozensets if directly hashing)
        # For robustness, convert item to a string representation for consistent hashing
        item_str = str(item)
        h = hashlib.sha256(f"{item_str}-{seed}".encode()).hexdigest()
        return int(h, 16) % self.num_bits

    def add(self, item):
        for i in range(self.num_hash_functions):
            index = self._hash(item, i)
            self.bit_array[index] = True

    def check(self, item) -> bool:
        for i in range(self.num_hash_functions):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True # Potentially in the set (false positive possible)