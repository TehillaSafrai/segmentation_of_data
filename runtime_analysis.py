import time
import numpy as np
import matplotlib.pyplot as plt
from segment import segment


# Function to measure runtime for the `segment` function
def measure_runtime_segment(x, p, q):
    start_time = time.time()
    _, cost = segment(x, p, q)
    end_time = time.time()
    return end_time - start_time  # Execution time in seconds


# Parameters for testing
sequence_lengths = [300, 400, 500, 700, 1000]  # Different lengths for `n`
penalty = 0.5
max_segment_lengths = [10, 30, 50, 100, 150]  # Different values for `q`

# 1. Runtime vs. Sequence Length
runtimes_by_length = []
for n in sequence_lengths:
    x = np.random.rand(n)  # Generate a random sequence of length n
    runtime = measure_runtime_segment(x, penalty, max_segment_lengths[1])  # Test with q=10
    runtimes_by_length.append(runtime)

# Plot: Runtime vs. Sequence Length
plt.plot(sequence_lengths, runtimes_by_length, marker='o')
plt.xlabel('Sequence Length (n)')
plt.ylabel('Runtime (s)')
plt.title('Segmentation Runtime vs. Sequence Length')
plt.show()

# 2. Runtime vs. Max Segment Length (q)
runtimes_by_max_len = []
for q in max_segment_lengths:
    x = np.random.rand(sequence_lengths[1])
    runtime = measure_runtime_segment(x, penalty, q)
    runtimes_by_max_len.append(runtime)

# Plot: Runtime vs. Max Segment Length
plt.plot(max_segment_lengths, runtimes_by_max_len, marker='o', color='orange')
plt.xlabel('Max Segment Length (q)')
plt.ylabel('Runtime (s)')
plt.title('Segmentation Runtime vs. Max Segment Length')
plt.show()

#Test with varying penalties (p)
penalties = [0.1, 0.5, 1.0, 2.0]
runtimes_by_penalty = []
for p in penalties:
    x = np.random.rand(sequence_lengths[1])
    runtime = measure_runtime_segment(x, p, max_segment_lengths[1])
    runtimes_by_penalty.append(runtime)

# Plot: Runtime vs. Penalty
plt.plot(penalties, runtimes_by_penalty, marker='o', color='green')
plt.xlabel('Penalty (p)')
plt.ylabel('Runtime (s)')
plt.title('Segmentation Runtime vs. Penalty')
plt.show()


def measure_runtime_segment_by_segments(x, p, q):
    """
    Measure the runtime of the `segment` function and count the number of segments produced.

    Args:
        x (np.ndarray): The input signal.
        p (float): Penalty parameter.
        q (int): Maximum segment length.

    Returns:
        tuple: (runtime, n_segments)
    """
    start_time = time.time()
    segments, _ = segment(x, p, q)
    runtime = time.time() - start_time
    n_segments = len(segments)
    return runtime, n_segments


runtimes = {}

for i in range(30):
    x = np.random.rand(300)  # Generate random sequence of length n
    runtime, k_segments = measure_runtime_segment_by_segments(x, penalty, 30)
    runtimes[k_segments] = runtime

# Plot runtime vs. number of segments
sorted_runtimes_by_key = dict(sorted(runtimes.items()))
plt.figure(figsize=(8, 6))
plt.plot(sorted_runtimes_by_key.items(), sorted_runtimes_by_key.values(), marker='o', color='purple')
plt.xlabel('Number of Segments')
plt.ylabel('Runtime (s)')
plt.title('Segmentation Runtime vs. Number of Segments')
plt.grid()
plt.show()


