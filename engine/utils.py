import tempfile
from urllib.request import urlopen
import resampy
import soundfile as sf
import requests

from vllm.multimodal.utils import fetch_video, fetch_image

__all__ = ["fetch_image"]

from typing import List, Dict, Union, Optional

def resample_wav_to_16khz(input_filepath):
    data, original_sample_rate = sf.read(input_filepath)
    # Only use the first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    # resample to 16kHz
    data_resampled = resampy.resample(data,
                                      sr_orig=original_sample_rate,
                                      sr_new=16000)
    return data_resampled

def fetch_and_read_video(video_url: str, fps=2):
  
    def read_video(video_file_name: Union[str, List[str]]):
        video, total_duration, nframes, second_per_grid = fetch_video(
            {'video': video_file_name})
        if total_duration is None and nframes is None:
            nframes = len(video)
            total_duration = 0.5 * nframes
       
        return video


    if isinstance(video_url, str) and video_url.startswith("http"):
        with tempfile.NamedTemporaryFile(delete=True) as temp_video_file:
            resp = requests.get(video_url)
            assert resp.status_code == requests.codes.ok, f"Failed to fetch video from {video_url}, status_code:{resp.status_code}, resp:{resp}"

            temp_video_file.write(urlopen(video_url).read())
            temp_video_file_path = temp_video_file.name
            video_file_name = temp_video_file_path
            return read_video(video_file_name)
    else:
        video_file_name = video_url
        return read_video(video_file_name)
