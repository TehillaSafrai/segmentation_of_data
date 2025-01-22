# Computational Biology Assignment - Sequence Alignment and Data Segmentation
**Due Date: November 17, 2024**

**Data Segmentation Algorithm**
Implement a dynamic programming solution for optimal segmentation of a data sequence into continuous segments.

### Implementation Requirements
The program should implement the following function:
```python
segs, cost = segment(x, p, q)
```
where:
- `x`: Input sequence
- `p`: Penalty parameter for regularization
- `q`: Maximum segment length
- Returns: 
  - `segs`: k x 3 table with start, end, and mean value for each segment
  - `cost`: Total score of the segmentation

### Command Line Usage
```bash
python3 segment.py --filepath in.txt --penalty 0.5 --maxlen 10
```

### Output Format
The program should output each segment on a new line in the format:
```
start_pos end_pos mean_value
```
followed by the total score on the last line.

Example:
```
1 9 0.945
10 22 0.631
23 29 0.953
...
40.238
```

### Technical Requirements
- All numerical values should be printed with 3 decimal places
- Runtime should be efficient (seconds or less for thousands of bases)
- Clear code documentation and explanations required
