# initiate test for setting seed 
import unittest
from atomsci.ddm.pipeline import random_seed as rs  

#--------------------------------------------------------------------------------------

class TestSeedMethods(unittest.TestCase):

    def test_generate_new_seed(self):
        """
        Test case to verify the generation of a new seed and random state. 
        
        This test checks:
        1. Generation of a new seed using generate_new_seed().
        2. Comparison of the generated seed with the global seed obtained via get_seed().
        3. Comparison of the generated random state with the global random state obtained via get_random_state().
        4. Generation of another new seed and random state to ensure they are different from the previous ones.
        5. Comparison of the second generated seed with the global seed obtained via get_seed().
        6. Comparison of the second generated random state with the global random state obtained via get_random_state().
        7. Verification that the random states generated are different for different seeds.

        This ensures that the seed generation and setting functions work correctly and consistently.
        """
        # Generate a new seed and check if it is set correctly
        seed1, random_state1 = rs.generate_new_seed()
        self.assertIsNotNone(seed1)
        self.assertEqual(seed1, rs.get_seed())
        self.assertEqual(random_state1, rs.get_random_state())

        # Generate another new seed and ensure it's different
        seed2, random_state2 = rs.generate_new_seed()
        self.assertIsNotNone(seed2)
        self.assertNotEqual(seed1, seed2)
        self.assertEqual(seed2, rs.get_seed())
        self.assertEqual(random_state2, rs.get_random_state())
        self.assertNotEqual(random_state1, random_state2)

    def test_set_seed(self):
        """
        Test case to verify the setting of a specific seed.

        This test checks:
        1. Setting a specific seed using set_seed().
        2. Comparison of the set seed with the global seed obtained via get_seed().
        3. Generation of random numbers using the set seed and comparison with expected values.

        This ensures that setting a specific seed results in consistent random number generation.
        """
        # Set a specific seed and check if it is set correctly
        rs.set_seed(42)
        self.assertEqual(rs.get_seed(), 42)
        random_state = rs.get_random_state()
        numbers = [random_state.random() for _ in range(5)]
        expected_numbers = [0.7739560485559633, 0.4388784397520523, 0.8585979199113825, 0.6973680290593639, 0.09417734788764953]
        self.assertEqual(numbers, expected_numbers)

    def test_random_state_consistency(self):
        """
        Test case to verify the consistency of random state with the same seed.

        This test checks:
        1. Setting a specific seed using set_seed().
        2. Generation of random numbers using the set seed and saving the random state.
        3. Resetting the same seed and generating random numbers again.
        4. Comparison of the random numbers generated from the same seed to ensure consistency.

        This ensures that the random state is consistent when the same seed is set multiple times.
        """
        # Set a specific seed and generate random numbers
        rs.set_seed(42)
        random_state1 = rs.get_random_state()
        numbers1 = [random_state1.random() for _ in range(5)]

        # Reset the same seed and generate random numbers again
        rs.set_seed(42)
        random_state2 = rs.get_random_state()
        numbers2 = [random_state2.random() for _ in range(5)]

        # Check if the random numbers are the same
        self.assertEqual(numbers1, numbers2)

    def test_random_state_different_seeds(self):
        """
        Test case to verify the difference in random state with different seeds.

        This test checks:
        1. Setting a specific seed using set_seed() and generating random numbers.
        2. Setting a different seed using set_seed() and generating random numbers again.
        3. Comparison of the random numbers generated from different seeds to ensure they are different.

        This ensures that different seeds produce different random states and sequences of random numbers.
        """
        # Set a specific seed and generate random numbers
        rs.set_seed(42)
        random_state1 = rs.get_random_state()
        numbers1 = [random_state1.random() for _ in range(5)]

        # Set a different seed and generate random numbers
        rs.set_seed(12345)
        random_state2 = rs.get_random_state()
        numbers2 = [random_state2.random() for _ in range(5)]

        # Check if the random numbers are different
        self.assertNotEqual(numbers1, numbers2)

if __name__ == '__main__':
    unittest.main()