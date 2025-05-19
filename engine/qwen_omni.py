from vllm import AsyncLLMEngine, SamplingParams

from vllm.engine.async_llm_engine import AsyncEngineArgs
from vllm.inputs import TextPrompt
from transformers import AutoProcessor, AutoTokenizer

from loguru import logger
import time

from typing import List, Dict, Union, Optional, AsyncGenerator, Tuple
from engine.utils import (
    resample_wav_to_16khz,
    fetch_and_read_video,
    fetch_image
)
import tempfile
import librosa
from urllib.request import urlopen
from llm_engine import LLMEngine
import os

os.environ["VLLM_USE_V1"] = '0'
class QwenOmniLLMEngine(LLMEngine):
    def __init__(self, model_name: str):
        self._max_model_len = 5632
        self._max_num_seqs = 5
        self._limit_mm_per_prompt = {
            "audio": 5,
            "image": 5,
            "video": 0,
        }
        self._model_name = model_name
        self._ready = False

    def _initialize_engine(self):
        """Initialize the vLLM engine for inference."""
        engine_args = AsyncEngineArgs(
            model=self._model_name,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            limit_mm_per_prompt = self._limit_mm_per_prompt,
            trust_remote_code=True,
        )
        
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    def _initialize_tokenizer(self):
        """Initialize the tokenizer for the model."""
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._processor = AutoProcessor.from_pretrained(self._model_name)

    def start(self):
        """Start the LLM engine and tokenizer."""
        try:
            self._initialize_engine()
            self._initialize_tokenizer()
            self._ready = True
            logger.info(f"LLM engine {self._model_name} is ready.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM engine: {e}")
            self._ready = False
            raise e

    async def generate_response(self, messages: list[Dict[str, Union[str, List[Dict[str, str]]]]], temperature=0.2, max_tokens=4096, request_id=str(time.time())) -> AsyncGenerator[Tuple[str, bool], bool]:
        if not self._ready:
            logger.error("LLM engine is not ready.")
            return
    
        text_prompt = self.make_inputs_qwen2_omni(
            messages=messages,
            use_audio_in_video=False,
        )
        logger.info(f"Text prompt: {text_prompt}")
        sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_tokens
        )
        outputs = self._engine.generate(text_prompt,
                        sampling_params=sampling_params, request_id=request_id)
        
        fist_time = True
        prev_text = ""
        async for o in outputs:
            generated_text = o.outputs[0].text
            finnished = o.finished
            delta = ""
            if generated_text.startswith(prev_text):
                delta = generated_text[len(prev_text):]
           
            prev_text = generated_text
            if delta:
                if fist_time:
                    logger.info(f"First time: {delta}")
                    fist_time = False
                yield (delta, finnished)


    def make_inputs_qwen2_omni(
        self,
        messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
        use_audio_in_video: Optional[bool] = False,
    ) -> TextPrompt:
      
        audios, images, videos = [], [], []
        for message in messages:
            if not isinstance(message['content'], list):
                message['content'] = [{
                    'type': 'text',
                    'text': message['content'],
                }]
            index, num_contents = 0, len(message['content'])
            while index < num_contents:
                ele = message['content'][index]
                if 'type' not in ele:
                    if 'text' in ele:
                        ele['type'] = 'text'
                    elif 'audio' in ele:
                        ele['type'] = 'audio'
                    elif 'audio_url' in ele:
                        ele['type'] = 'audio_url'
                    elif 'image' in ele:
                        ele['type'] = 'image'
                    elif 'image_url' in ele:
                        ele['type'] = 'image_url'
                    elif 'video' in ele:
                        ele['type'] = 'video'
                    elif 'video_url' in ele:
                        ele['type'] = 'video_url'
                    else:
                        raise ValueError(f'Unknown ele: {ele}')

                if ele['type'] == 'audio' or ele['type'] == 'input_audio' or ele['type'] == 'audio_url':
                    if 'audio_url' in ele:
                        audio_key = 'audio_url'
                        with tempfile.NamedTemporaryFile(
                                delete=True) as temp_audio_file:
                            temp_audio_file.write(urlopen(ele[audio_key]).read())
                            temp_audio_file_path = temp_audio_file.name
                            audios.append(
                                resample_wav_to_16khz(temp_audio_file_path))
                            ele['audio'] = temp_audio_file_path
                    elif 'audio' in ele:
                        audio_key = 'audio'
                        audios.append(ele[audio_key])
                    else:
                        raise ValueError(f'Unknown ele {ele}')
                elif use_audio_in_video and (ele['type'] == 'video'
                                            or ele['type'] == 'video_url'):
                    # use video as audio as well
                    if 'video_url' in ele:
                        audio_key = 'video_url'
                        with tempfile.NamedTemporaryFile(
                                delete=True) as temp_video_file:
                            temp_video_file.write(urlopen(ele[audio_key]).read())
                            temp_video_file_path = temp_video_file.name
                            ele[audio_key] = temp_video_file_path
                            audios.append(
                                librosa.load(temp_video_file_path, sr=16000)[0])
                            videos.append(
                                fetch_and_read_video(temp_video_file_path))
                            ele['video'] = temp_video_file_path
                    elif 'video' in ele:
                        audio_key = 'video'
                        audios.append(librosa.load(ele[audio_key], sr=16000)[0])
                        videos.append(fetch_and_read_video(ele[audio_key]))
                    else:
                        raise ValueError("Unknown ele {}".format(ele))
                    # insert a audio after the video
                    message['content'].insert(index + 1, {
                        "type": "audio",
                        "audio": ele[audio_key],
                    })
                    # no need to load the added audio again
                    index += 1
                elif ele['type'] == 'video' or ele['type'] == 'video_url':
                    if 'video_url' in ele:
                        video_key = 'video_url'
                        with tempfile.NamedTemporaryFile(
                                delete=True) as temp_video_file:
                            temp_video_file.write(urlopen(ele['video_url']).read())
                            temp_video_file_path = temp_video_file.name
                            videos.append(fetch_and_read_video(temp_video_file))
                            ele['video'] = temp_video_file_path
                    else:
                        video_key = 'video'
                        videos.append(fetch_and_read_video(ele[video_key]))
                elif ele['type'] == 'image' or ele['type'] == 'image_url':
                    images.append(fetch_image(ele))

                # move to the next content
                index += 1

        prompt = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            add_vision_id=True,
        )

        audios = audios if len(audios) > 0 else None
        images = images if len(images) > 0 else None
        videos = videos if len(videos) > 0 else None

        logger.info(f'{prompt}, '
                    f'audios = {len(audios) if audios else None}, '
                    f'images = {len(images) if images else None}, '
                    f'videos = {len(videos) if videos else None}')

        multi_modal_data = {}
        if audios:
            multi_modal_data["audio"] = audios
        if images:
            multi_modal_data["image"] = images
        if videos:
            multi_modal_data["video"] = videos
            # pass through the use_audio_in_video to llm engine
            multi_modal_data["use_audio_in_video"] = use_audio_in_video


        if isinstance(prompt, list) and isinstance(prompt[0], (list, str)):
            prompt = prompt[0]

        return TextPrompt(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
        )