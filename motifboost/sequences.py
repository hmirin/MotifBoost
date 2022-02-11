try:
    from typing import Dict, Final, List, Optional, Union
except:
    from typing import Dict, List, Optional, Union
    from typing_extensions import Final


import numpy as np
from bitarray import bitarray


class PackedStringArray:
    def __init__(self, alphabets: List[str]):
        assert all([len(ch) == 1 for ch in alphabets])
        self.alphabets: Final[List[str]] = alphabets
        self.alphabet_size: Final[int] = len(self.alphabets)
        self.char_bit_size: Final[int] = int(np.ceil(np.log2(self.alphabet_size)))
        self.dictionary: Final[Dict[str, bitarray]] = {
            ch: bitarray(bin(idx)[2:].zfill(self.char_bit_size))
            for idx, ch in enumerate(alphabets)
        }
        self.data: Final[bitarray] = bitarray()
        self.indices: np.ndarray = np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getbytes__(self, idx) -> bitarray:
        if idx + 1 == len(self.indices):
            word = self.data[self.indices[idx] :]
        else:
            word = self.data[self.indices[idx] : self.indices[idx + 1]]
        return word

    def __getitem__(self, idx) -> str:
        return "".join(self.__getbytes__(idx).decode(self.dictionary))

    def append(self, word: str):
        self.indices = np.append(self.indices, len(self.data))
        self.data.encode(self.dictionary, word)

    def bulk_append(self, words: List[str]):
        word = "".join(words)
        indices = np.cumsum(
            [len(self.data)]
            + [len(w) * self.char_bit_size for idx, w in enumerate(words)][:-1]
        )
        self.data.encode(self.dictionary, word)
        self.indices = np.concatenate((self.indices, indices))

    def get_all_strs(self) -> List[str]:
        return [self[i] for i in range(len(self))]

    def get_all_bytes(self) -> List[bitarray]:
        return [self.__getbytes__(idx) for idx in range(len(self))]


class SequenceContainer(object):
    data: Union[List[str], PackedStringArray]

    def __init__(
        self, alphabets: Optional[List[str]] = None, save_memory: bool = False
    ):
        self.alphabets = alphabets
        self.save_memory = save_memory
        if save_memory:
            self.data = PackedStringArray(alphabets)
        else:
            self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def get_all(self) -> List[str]:
        if isinstance(self.data, PackedStringArray):
            return self.data.get_all_strs()
        else:
            return self.data

    def bulk_append(self, words: List[str]):
        if isinstance(self.data, PackedStringArray):
            self.data.bulk_append(words)
        else:
            self.data = self.data + words

    def append(self, word: str):
        self.bulk_append([word])

    def __getitem__(self, idx) -> str:
        return self.data[idx]
