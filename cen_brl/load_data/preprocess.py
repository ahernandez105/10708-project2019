import numpy as np

def pad_sequences(sequences, maxlen=None, dtype='int32',
                padding='pre', truncating='pre', value=0.0):
    """
    Pad sequences to the same length, by truncating/padding
    """

    if isinstance(sequences, np.ndarray):
        sequences = sequences.tolist()

    new_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                new_seq = seq[-maxlen:]
            elif truncating == 'post':
                new_seq = seq[:maxlen]
            else:
                raise ValueError(truncating)

        else:
            diff = maxlen - len(seq)
            if padding == 'pre':
                new_seq = [value] * diff + seq
            elif padding == 'post':
                new_seq = seq + [value] * diff
            else:
                raise ValueError(padding)

        new_sequences.append(new_seq)

    return np.array(new_sequences, dtype=dtype)
