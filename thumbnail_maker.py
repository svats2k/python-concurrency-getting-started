# thumbnail_maker.py
import time
import rich
import re
import os
import logging
from tqdm import tqdm
from urllib.parse import urlparse
from urllib.request import urlretrieve
from pathlib import Path
import numpy as np

import PIL
from PIL import Image

from functools import wraps
from typing import Any, Callable, List

import logging

from rich.logging import RichHandler

logger = logging.getLogger(__name__)
shell_handler = RichHandler()

logger.setLevel(logging.DEBUG)
shell_handler.setLevel(logging.DEBUG)

# the formatter determines how the logger looks like
FMT_SHELL = "%(message)s"
FMT_FILE = """%(levelname)s %(asctime)s [%(filename)s
    %(funcName)s %(lineno)d] %(message)s"""

shell_formatter = logging.Formatter(FMT_SHELL)
shell_handler.setFormatter(shell_formatter)
logger.addHandler(shell_handler)

def timeit(func: Callable) -> Callable:
    @wraps(func)
    def timed_func(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Executed {func.__name__} in {end_time-start_time:.3f}")
        return result
    
    return timed_func

class ThumbnailMakerService(object):
    def __init__(self, home_dir:str='.') -> None:
        self.home_dir = Path(home_dir)
        self.input_dir:Path = self.home_dir/'incoming'
        self.output_dir:Path = self.home_dir/'outgoing'

    @timeit
    def get_images(self, img_list: List[str]):
        # validate inputs
        if not img_list:
            return
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        pbar = tqdm(img_list, leave=True)
        for img_loc in pbar:
            pbar.set_description(desc=Path(img_loc).name)
            if not (self.input_dir/Path(img_loc).name).exists:
                if re.search(pattern='https', string=img_loc):
                    img_filename = urlparse(img_loc).path.split('/')[-1]
                    urlretrieve(img_loc, self.input_dir/img_filename)

        logger.info(f"Image : {len(list(self.input_dir.glob('*')))} already present")


    @timeit
    def perform_resizing(self) -> None:
        # validate inputs
        if not self.input_dir.exists():
            logger.info("Input directory missing ..")
            return
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

        target_sizes = [32, 64, 200]
        img_files = self.input_dir.glob("*")

        for filename in tqdm(img_files):
            print(f"file name: {str(filename)}")
            orig_img = Image.open(filename)
            logger.info(F"{filename.name}: {np.asarray(orig_img).shape}")
            for basewidth in target_sizes:
                img = orig_img
                # calculate target height of the resized image to maintain the aspect ratio
                wpercent = (basewidth / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                # perform resizing
                img = img.resize((basewidth, hsize), PIL.Image.LANCZOS)
                
                # save the resized image to the output dir with a modified file name 
                new_filename = self.output_dir/f"{filename.stem}_{str(basewidth)}{filename.suffix}"
                img.save(new_filename)

    @timeit
    def make_thumbnails(self, img_url_list: List[str]) -> None:

        self.get_images(img_url_list)
        self.perform_resizing()
