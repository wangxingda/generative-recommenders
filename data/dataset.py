# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple
import pdb
import pandas as pd
import torch
import glob
import dask.dataframe as dd
class DatasetV2(torch.utils.data.Dataset):
    """In reverse chronological order."""

    def __init__(
        self,
        ratings_file: str,
        padding_length: int,
        ignore_last_n: int,  # used for creating train/valid/test sets
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
    ) -> None:
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        super().__init__()

        self.ratings_frame = pd.read_csv(
            ratings_file,
            delimiter=",",
            # iterator=True,
        )
        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio

    def __len__(self) -> int:
        return len(self.ratings_frame)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if idx in self._cache.keys():
            return self._cache[idx]
        sample = self.load_item(idx)
        self._cache[idx] = sample
        return sample

    def load_item(self, idx) -> Dict[str, torch.Tensor]:
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        def eval_as_list(x, ignore_last_n) -> List[int]:
            y = eval(x)
            y_list = [y] if type(y) == int else list(y)
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def eval_int_list(
            x,
            target_len: int,
            ignore_last_n: int,
            shift_id_by: int,
            sampling_kept_mask: Optional[List[bool]],
        ) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        if self._sample_ratio < 1.0:
            raw_length = len(eval_as_list(data.sequence_item_ids, self._ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        movie_history, movie_history_len = eval_int_list(
            data.sequence_item_ids,
            self._padding_length,
            self._ignore_last_n,
            shift_id_by=self._shift_id_by,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_history_ratings, ratings_len = eval_int_list(
            data.sequence_ratings,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_timestamps, timestamps_len = eval_int_list(
            data.sequence_timestamps,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        assert (
            movie_history_len == timestamps_len
        ), f"history len {movie_history_len} differs from timestamp len {timestamps_len}."
        assert (
            movie_history_len == ratings_len
        ), f"history len {movie_history_len} differs from ratings len {ratings_len}."

        def _truncate_or_pad_seq(
            y: List[int], target_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == target_len
            return y

        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]
        if self._chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()

        max_seq_len = self._padding_length - 1
        history_length = min(len(historical_ids), max_seq_len)
        historical_ids = _truncate_or_pad_seq(
            historical_ids,
            max_seq_len,
            self._chronological,
        )
        historical_ratings = _truncate_or_pad_seq(
            historical_ratings,
            max_seq_len,
            self._chronological,
        )
        historical_timestamps = _truncate_or_pad_seq(
            historical_timestamps,
            max_seq_len,
            self._chronological,
        )
        # moved to features.py
        # if self._chronological:
        #     historical_ids.append(0)
        #     historical_ratings.append(0)
        #     historical_timestamps.append(0)
        # print(historical_ids, historical_ratings, historical_timestamps, target_ids, target_ratings, target_timestamps)
        ret = {
            "user_id": user_id,
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(
                historical_timestamps, dtype=torch.int64
            ),
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
        }
        return ret

    
# class DatasetV3(torch.utils.data.Dataset):
#     """In reverse chronological order."""

#     def __init__(
#         self,
#         ratings_file: str,
#         padding_length: int,
#         ignore_last_n: int,  # used for creating train/valid/test sets
#         shift_id_by: int = 0,
#         chronological: bool = False,
#     ) -> None:
#         """
#         Args:
#             csv_file (string): Path to the csv file.
#         """
#         super().__init__()
# #         ratings_files = glob.glob(ratings_file)
# #         self.ratings_frame = pd.concat(
# #             [pd.read_csv(file, delimiter=",", header=0) for file in ratings_files],
# #             ignore_index=True
# #         )
        
#         ratings_files = glob.glob(ratings_file)
#         self.ratings_frame = pd.concat(
#             [pd.read_csv(file, delimiter="\t", names=['uid', 'sequence_item_ids', 'whethe_click', 'sequence_timestamps']) for file in ratings_files],
#             ignore_index=True
#         )
        
# #         print(len(self.ratings_frame))
# #         pdb.set_trace()
        
# #         self.ratings_frame = pd.read_csv(ratings_file, delimiter=",", header=0) 
# #         ddf = dd.read_csv(ratings_file, delimiter=",", header=0)
# #         self.ratings_frame = ddf.compute()
# #         self.ratings_frame = pd.read_csv(
# #             ratings_file,
# #             delimiter=",",
# #             # iterator=True,
# #         )
#         self._padding_length: int = padding_length
#         self._ignore_last_n: int = ignore_last_n
#         self._cache = dict()
#         self._shift_id_by: int = shift_id_by
#         self._chronological: bool = chronological
            

#     def __len__(self) -> int:
#         return len(self.ratings_frame)

#     def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
#         if idx in self._cache.keys():
#             return self._cache[idx]
#         sample = self.load_item(idx)
#         self._cache[idx] = sample
#         return sample
    

#     def load_item(self, idx) -> Dict[str, torch.Tensor]:
#         data = self.ratings_frame.iloc[idx]

#         def find_last_index_of_one(seq):
#             count = 0
#             for i in range(len(seq) - 1, -1, -1):
#                 if seq[i] == 1:
#                     last_one = i
#                     return last_one
            
#         def eval_as_list(x, ignore_last_n) -> List[int]:
#             y = eval(x)
#             y_list = [y] if type(y) == int else list(y)
# #             if ignore_last_n > 0:
# #                 # for training data creation
# #                 y_list = y_list[:-ignore_last_n]
#             return y_list

#         def eval_int_list(
#             x,
#             target_len: int,
#             ignore_last_n: int,
#             shift_id_by: int,
#             sampling_kept_mask: Optional[List[bool]],
#         ) -> Tuple[List[int], int]:
#             y = eval_as_list(x, ignore_last_n=ignore_last_n)
#             if sampling_kept_mask is not None:
#                 y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
#             y_len = len(y)
#             if shift_id_by > 0:
#                 y = [x + shift_id_by for x in y]
#             return y, y_len

#         sampling_kept_mask = None
#         movie_history, movie_history_len = eval_int_list(
#             data.sequence_item_ids,
#             self._padding_length,
#             self._ignore_last_n,
#             shift_id_by=self._shift_id_by,
#             sampling_kept_mask=sampling_kept_mask,
#         )
#         movie_history_ratings, ratings_len = eval_int_list(
#             data.whethe_click,
#             self._padding_length,
#             self._ignore_last_n,
#             0,
#             sampling_kept_mask=sampling_kept_mask,
#         )
#         movie_timestamps, timestamps_len = eval_int_list(
#             data.sequence_timestamps,
#             self._padding_length,
#             self._ignore_last_n,
#             0,
#             sampling_kept_mask=sampling_kept_mask,
#         )
        
#         last_index = find_last_index_of_one(movie_history_ratings)
        
#         movie_history = movie_history[:last_index + 1]
#         movie_history.reverse()
#         movie_history_len = len(movie_history)
        
#         movie_history_ratings = movie_history_ratings[:last_index + 1]
#         movie_history_ratings.reverse()
#         ratings_len = len(movie_history_ratings)
        
#         movie_timestamps = movie_timestamps[:last_index + 1]
#         movie_timestamps.reverse()
#         timestamps_len = len(movie_timestamps)
#         assert movie_history_len == ratings_len ==timestamps_len
        
#         assert (
#             movie_history_len == timestamps_len
#         ), f"history len {movie_history_len} differs from timestamp len {timestamps_len}."
#         assert (
#             movie_history_len == ratings_len
#         ), f"history len {movie_history_len} differs from ratings len {ratings_len}."

#         def _truncate_or_pad_seq(
#             y: List[int], target_len: int, chronological: bool
#         ) -> List[int]:
#             y_len = len(y)
#             if y_len < target_len:
#                 y = y + [0] * (target_len - y_len)
#             else:
#                 if not chronological:
#                     y = y[:target_len]
#                 else:
#                     y = y[-target_len:]
#             assert len(y) == target_len
#             return y

#         historical_ids = movie_history[1:]
#         historical_ratings = movie_history_ratings[1:]
#         historical_timestamps = movie_timestamps[1:]
#         target_ids = movie_history[0]
#         target_ratings = movie_history_ratings[0]
#         target_timestamps = movie_timestamps[0]
#         if self._chronological:
#             historical_ids.reverse()
#             historical_ratings.reverse()
#             historical_timestamps.reverse()

#         max_seq_len = self._padding_length - 1
#         history_length = min(len(historical_ids), max_seq_len)
#         historical_ids = _truncate_or_pad_seq(
#             historical_ids,
#             max_seq_len,
#             self._chronological,
#         )
#         historical_ratings = _truncate_or_pad_seq(
#             historical_ratings,
#             max_seq_len,
#             self._chronological,
#         )
#         historical_timestamps = _truncate_or_pad_seq(
#             historical_timestamps,
#             max_seq_len,
#             self._chronological,
#         )
#         ret = {
#             "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
#             "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
#             "historical_timestamps": torch.tensor(
#                 historical_timestamps, dtype=torch.int64
#             ),
#             "history_lengths": history_length,
#             "target_ids": target_ids,
#             "target_ratings": target_ratings,
#             "target_timestamps": target_timestamps,
#         }
#         return ret
    

class DatasetV3(torch.utils.data.IterableDataset):
    def __init__(
        self,
        ratings_file: str,
        padding_length: int,
        ignore_last_n: int,  # used for creating train/valid/test sets
        shift_id_by: int = 0,
        chronological: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()

        self.ratings_files = ratings_file
        self._padding_length = padding_length
        self._ignore_last_n = ignore_last_n
        self._shift_id_by = shift_id_by
        self._chronological = chronological
        
        self.rank = rank
        self.world_size = world_size
        
        self.start = 0
        self.end = len(self.ratings_files)

    def __iter__(self):
        for file in self.ratings_files[self.start:self.end]:
#             if self.rank == 1 :
#                 print(file)
            for chunk in pd.read_csv(file, delimiter="\t", names=['uid', 'sequence_item_ids', 'whethe_click', 'sequence_timestamps'], chunksize=500000):
                chunk = chunk.iloc[self.rank::self.world_size]
#                 if self.rank == 1 :
#                     print(len(chunk))
                for idx in range(len(chunk)):
                    data = chunk.iloc[idx]
                    try:
                        yield self.load_item(data)
                    except Exception as e:
                        pass
#                         print(f"Skipping invalid data at index {idx} in file {file}. Error: {e}")
#                         print(f"Invalid data: {data}")

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        def find_last_index_of_one(seq):
            count = 0
            for i in range(len(seq) - 1, -1, -1):
                if seq[i] == 1:
                    last_one = i
                    return last_one

        def eval_as_list(x, ignore_last_n) -> List[int]:
            y = eval(x)
            y_list = [y] if type(y) == int else list(y)
            return y_list

        def eval_int_list(
            x,
            target_len: int,
            ignore_last_n: int,
            shift_id_by: int,
            sampling_kept_mask: Optional[List[bool]],
        ) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        sampling_kept_mask = None
        movie_history, movie_history_len = eval_int_list(
            data.sequence_item_ids,
            self._padding_length,
            self._ignore_last_n,
            shift_id_by=self._shift_id_by,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_history_ratings, ratings_len = eval_int_list(
            data.whethe_click,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_timestamps, timestamps_len = eval_int_list(
            data.sequence_timestamps,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        
        last_index = find_last_index_of_one(movie_history_ratings)
        
        movie_history = movie_history[:last_index + 1]
        movie_history.reverse()
        movie_history_len = len(movie_history)
        
        movie_history_ratings = movie_history_ratings[:last_index + 1]
        movie_history_ratings.reverse()
        ratings_len = len(movie_history_ratings)
        
        movie_timestamps = movie_timestamps[:last_index + 1]
        movie_timestamps.reverse()
        timestamps_len = len(movie_timestamps)
        assert movie_history_len == ratings_len ==timestamps_len
        
        assert (
            movie_history_len == timestamps_len
        ), f"history len {movie_history_len} differs from timestamp len {timestamps_len}."
        assert (
            movie_history_len == ratings_len
        ), f"history len {movie_history_len} differs from ratings len {ratings_len}."

        def _truncate_or_pad_seq(
            y: List[int], target_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == target_len
            return y

        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]
        if self._chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()

        max_seq_len = self._padding_length - 1
        history_length = min(len(historical_ids), max_seq_len)
        historical_ids = _truncate_or_pad_seq(
            historical_ids,
            max_seq_len,
            self._chronological,
        )
        historical_ratings = _truncate_or_pad_seq(
            historical_ratings,
            max_seq_len,
            self._chronological,
        )
        historical_timestamps = _truncate_or_pad_seq(
            historical_timestamps,
            max_seq_len,
            self._chronological,
        )
        ret = {
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(
                historical_timestamps, dtype=torch.int64
            ),
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
        }
        return ret
    
