import io
import os
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import portalocker
import torch
from PIL import Image

from ..utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def info_memory():
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    summary = torch.cuda.memory_summary()
    return f" Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB\n{summary}"


def get_indices_from_tokens(token_by_word: Dict[int, str], word_list: List[str]) -> List[List[int]]:
    result = []
    max_size = len(token_by_word)

    for entity_word in word_list:
        # Split composed words into individual words
        composed_words = entity_word.split()
        if len(composed_words) > 1:
            # Handle composed words
            temp_result = []
            for sub_word in composed_words:
                sub_result = get_indices_from_tokens(token_by_word, [sub_word])
                if sub_result:
                    temp_result.extend(sub_result[0])  # Merge indices for the composed word
                else:
                    break
            if temp_result:
                result.append(temp_result)
        else:
            # Handle single words
            for idx, word_token in token_by_word.items():
                if word_token and entity_word.startswith(word_token):
                    temp_list = [idx]
                    entity_word_left = entity_word[len(word_token) :]
                    if entity_word_left == "":
                        result.append(temp_list)
                        break
                    for idx_ in range(idx + 1, max_size):
                        if entity_word_left.startswith(token_by_word[idx_]):
                            temp_list.append(idx_)
                            entity_word_left = entity_word_left[len(token_by_word[idx_]) :]
                            if not entity_word_left:
                                result.append(temp_list)
                                break
                        else:
                            break
    return result


class ListHydra(list):
    def __init__(self, start, stop, step=1):
        super().__init__(range(start, stop, step))


def get_object_list(params):
    list_objects = []
    color_objects = []

    for k, v in params.get("labels_params", {}).items():
        if k.startswith("object") or k.startswith("entity"):
            while len(list_objects) < len(v):
                list_objects.append([])
            for i, v_ in enumerate(v):
                list_objects[i].append(v_)

    for k, v in params.get("adjs_params", {}).items():
        if k.startswith("adj"):
            while len(color_objects) < len(v):
                color_objects.append([])
            for i, v_ in enumerate(v):
                color_objects[i].append(v_)

    return list_objects, color_objects


class TypeData:
    def __init__(self, type_data: str):
        if type_data == "image":
            self.type_data = "image"
        elif type_data == "text":
            self.type_data = "txt"
        elif type_data == "json":
            self.type_data = "json"
        else:
            raise ValueError(f"Type data {type_data} not recognized")

    def is_image(self):
        return self.type_data == "image"

    def is_text(self):
        return self.type_data == "txt"

    def is_json(self):
        return self.type_data == "json"


EXTENSIONS_IMAGE = [".jpg", ".jpeg", ".png"]


class ConcurrentWriter:
    def __init__(self, path: str | Path, type_data: str):
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.type_data = TypeData(type_data)
        self._init_storage()

    def _lock_file(self):
        self.file = open(f"{self.path}.lock", "w")
        portalocker.lock(self.file, portalocker.LOCK_EX)

    def _unlock_file(self):
        portalocker.unlock(self.file)
        self.file.close()

    def _init_storage(self):
        """Initialize storage based on type"""
        if self.type_data.is_text():
            # Check if file exists, create if not
            if not os.path.exists(self.path):
                with portalocker.Lock(f"{self.path}.lock"):
                    with open(self.path, "w") as f:
                        f.write("")
        elif self.type_data.is_json():
            with portalocker.Lock(f"{self.path}.lock"):
                with tarfile.open(self.path, "w:") as _:
                    pass
        elif self.type_data.is_image():
            # Create directory if it doesn't exist
            os.makedirs(self.path, exist_ok=True)
        else:
            raise ValueError(f"Type {self.type_data} not recognized")

    def _open_tarfile(self, mode, timeout=5):
        """Safely open a tarfile with retries and error handling"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return tarfile.open(self.path, mode)
            except tarfile.ReadError as e:
                if attempt < max_retries - 1:
                    # Small delay before retry
                    time.sleep(timeout)
                else:
                    raise ValueError(f"Failed to open tarfile after {max_retries} attempts: {e}")

    def write_text(self, line: str):
        """Append text line with locking"""
        if not self.type_data.is_text():
            raise ValueError("Writer not initialized for text")

        with portalocker.Lock(f"{self.path}.lock"):
            with open(self.path, "a") as f:
                f.write(line + "\n")

    def write_images(self, images: List[Tuple[str, Image.Image]]):
        """Write list of (name, image) tuples to tar"""
        if not self.type_data.is_image():
            raise ValueError("Writer not initialized for images")

        for name, image in images:
            image.save(self.path / name)
            logger.info(f"Image {name} written to folder {self.path}")

    def write(self, **kwargs):
        if self.type_data.is_text():
            self.write_text(**kwargs)
        elif self.type_data.is_image():
            self.write_images(**kwargs)
        elif self.type_data.is_json():
            self.write_json(**kwargs)

    def __call__(self, *args, **kwds):
        self.write(*args, **kwds)

    def write_json(self, data: dict, name: str):
        """Write DataFrame as JSON to tar"""
        if not self.type_data.is_json():
            raise ValueError("Writer not initialized for JSON")
        df = pd.DataFrame(data)
        with portalocker.Lock(f"{self.path}.lock"):
            with self._open_tarfile("a:") as tar:
                # Convert to JSON string (orient='records' for consistent reading)
                json_data = df.to_json(orient="records").encode("utf-8")
                json_file = io.BytesIO(json_data)
                tarinfo = tarfile.TarInfo(name=name)
                tarinfo.size = len(json_data)
                tar.addfile(tarinfo, json_file)

    def read_all_lines(self) -> List[str]:
        """Read all lines from text file with locking"""
        if not self.type_data.is_text():
            raise ValueError("Writer not initialized for text")

        if not os.path.exists(self.path):
            return []
        with portalocker.Lock(f"{self.path}.lock"):
            with open(self.path, "r") as f:
                lines = f.readlines()
                # Remove empty lines and strip whitespace
                return [line.strip() for line in lines if line.strip()]

    def read_all_json(self) -> pd.DataFrame:
        """Read and concatenate all JSON files from tar into a DataFrame

        Returns:
            pd.DataFrame: Combined DataFrame from all JSON files
            Empty DataFrame if no files or path doesn't exist
        """
        if not self.type_data.is_json():
            raise ValueError("Writer not initialized for JSON")

        if not os.path.exists(self.path):
            return pd.DataFrame()

        all_dfs = []
        with portalocker.Lock(f"{self.path}.lock"):
            try:
                with self._open_tarfile("r") as tar:
                    all_members = tar.getmembers()
                    # remove duplicate
                    unique_members = {member.name: member for member in all_members}.values()

                    for member in unique_members:
                        if member.name.endswith(".json"):
                            f = tar.extractfile(member)
                            if f is not None:
                                try:
                                    content = f.read().decode("utf-8")
                                    df = pd.read_json(io.StringIO(content), orient="records")
                                    all_dfs.append(df)
                                finally:
                                    f.close()
            except (tarfile.ReadError, EOFError) as e:
                # Log the error but return empty dataframe
                print(f"Error reading tarfile: {e}")
                return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=False) if all_dfs else pd.DataFrame()

    def get_images_(self, names: List[str]) -> List[Image.Image]:
        """Extract multiple images from tar file.

        Args:
            names: List of image filenames to extract

        Returns:
            List of PIL Image objects

        Raises:
            ValueError: If writer not initialized for images
            KeyError: If image file not found in tar
        """
        if not self.type_data.is_image():
            raise ValueError("Writer not initialized for images")

        images = []

        for name in names:
            image_path = Path(self.path) / name
            if not image_path.exists():
                raise KeyError(f"Image `{name}` not found in folder {self.path}")
            image = Image.open(image_path)
            images.append(image)

        return images
