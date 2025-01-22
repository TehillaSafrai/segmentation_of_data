import sys

import numpy as np
import argparse


def read_signal_file(filepath: str) -> np.ndarray:
    """
    Read signal data from a tab-separated text file.
    Each line contains one or more columns of values separated by tabs.

    Args:
        filepath: Path to the input text file

    Returns:
        np.ndarray: Array of signal values from all columns
    """
    return np.loadtxt(filepath, delimiter='\t', dtype=np.float32)


def print_segments(s: np.ndarray, c: float) -> None:
    """
    Print segments array with 3 decimal places.
    Each row contains [start, end, average].

    Args:
        s: numpy array of shape (n_segments, 3)
        c: Total cost value of optimal segmentation
    """
    for row in s:
        print(f"{int(row[0])} {int(row[1])} {np.round(row[2], 3)}")
    print(np.round(c, 3))
    return


def create_segments_tabel(x: np.ndarray, p: float, q: int, score_tabel: np.ndarray):
    """
       Helper function to compute the minimum cost and the beginning index of the optimal segment.

       Args:
           x (np.ndarray): Current segment of the signal (1D array).
           p (float): Penalty parameter.
           q (int): Maximum segment length.
           score_tabel (np.ndarray): Array storing the minimum cost up to each point.

       Returns:
           tuple: (min_score, min_score_segments_beginning)
               - min_score (float): The minimum cost for the current segmentation.
               - min_score_segments_beginning (int): The starting index of the optimal segment.
       """
    min_score = sys.maxsize
    min_score_segments_beginning = 0
    for i in range(1, min(q, x.size)+1):
        last_i_elements = x[-i:]

        average = np.mean(last_i_elements)
        sum_of_squares = np.sum((last_i_elements - average) ** 2)

        if x.size - i - 1 >= 0:
            score = sum_of_squares + score_tabel[x.size - i - 1] + p
        else:
            score = sum_of_squares

        if score < min_score:
            min_score = score
            min_score_segments_beginning = x.size - i

    return min_score, min_score_segments_beginning


def segment(x: np.ndarray, p: float, q: int):
    """
    Segment a signal using dynamic programming.

    Args:
        x: Input signal (numpy 1d array)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, 3), each row in format [start, end, average]
        c: Total cost value of optimal segmentation
    """
    score_table = np.zeros(x.size)
    segments_tracker = np.zeros(x.size, dtype=int)

    for i in range(1, x.size+1):
        min_score, min_score_segments_beginning = create_segments_tabel(x[:i], p, q, score_table)
        score_table[i - 1] = min_score
        segments_tracker[i - 1] = min_score_segments_beginning

    s = np.empty((0, 3))
    end = x.size

    while end > 0:
        start = segments_tracker[end - 1]
        average = np.mean(x[start:end])
        s = np.vstack((np.array([start + 1, end, average]), s))
        end = start
    return s, score_table[-1]


def create_segments_tabel_multi_channel(x: np.ndarray, p: float, q: int, score_table: np.ndarray):
    """
        Helper function to compute the minimum cost and the beginning index of the optimal segment for multi-channel data.

        Args:
            x (np.ndarray): Current segment of the signal (2D array where rows are channels).
            p (float): Penalty parameter.
            q (int): Maximum segment length.
            score_table (np.ndarray): Array storing the minimum cost up to each point.

        Returns:
            tuple: (min_score, min_score_segments_beginning)
                - min_score (float): The minimum cost for the current segmentation.
                - min_score_segments_beginning (int): The starting index of the optimal segment.
        """
    min_score = sys.maxsize
    min_score_segments_beginning = 0

    for i in range(1, min(q, x.shape[1]) + 1):
        last_i_elements = x[:, -i:]
        averages = np.mean(last_i_elements, axis=1)

        sum_of_squares = np.sum((last_i_elements - averages[:, None]) ** 2)

        if x.shape[1] - i - 1 >= 0:
            score = sum_of_squares + score_table[x.shape[1] - i - 1] + p
        else:
            score = sum_of_squares

        if score < min_score:
            min_score = score
            min_score_segments_beginning = x.shape[1] - i

    return min_score, min_score_segments_beginning


def segment_multi_channel(x: np.ndarray, p: float, q: int):
    """
    Similar to segment but with multiple channels.

    Args:
        x: Input signal (numpy 2d array, where rows are different channels)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, 3), each row in format [start, end, average]
        c: Total cost value of optimal segmentation
    """
    score_table = np.zeros(x.shape[1])
    segments_tracker = np.zeros(x.shape[1], dtype=int)

    for i in range(1, x.shape[1] + 1):
        sub_array = x[:, :i]
        min_score, min_score_segments_beginning = create_segments_tabel_multi_channel(sub_array, p, q, score_table)
        score_table[i - 1] = min_score
        segments_tracker[i - 1] = min_score_segments_beginning

    # Initialize `s` with shape (0, 3) to ensure consistency in stacking
    s = np.empty((0, 3))
    end = x.shape[1]

    while end > 0:
        start = segments_tracker[end - 1]
        segment_data = x[:, start:end]
        avg_value = np.mean(segment_data)  # Average across all channels in the segment
        # Stack [start, end, avg_value] as a new row in `s`
        s = np.vstack((np.array([start + 1, end, avg_value]), s))
        end = start

    return s, score_table[-1]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process signal data from a text file.')
    parser.add_argument('--filepath', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--penalty', type=float, required=True, help='Penalty parameter')
    parser.add_argument('--max_len', type=int, required=True, help='Maximum segment length')
    parser.add_argument('--is_multi_channel', type=bool, required=False, default=False,
                        help='call segment_multi_channel')
    args = parser.parse_args()

    seq = read_signal_file(args.filepath)

    if args.is_multi_channel:
        segmented_output, c = segment_multi_channel(seq, args.penalty, args.max_len)
        print_segments(segmented_output, c)
    else:
        segmented_output, c = segment(seq, args.penalty, args.max_len)
        print_segments(segmented_output, c)
