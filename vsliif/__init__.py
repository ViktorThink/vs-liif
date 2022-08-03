import math
import os.path as osp

import numpy as np
import onnxruntime as ort
import vapoursynth as vs
import logging
from liif import process_image

import torch

dir_name = osp.dirname(__file__)


def RealESRGAN(
    clip: vs.VideoNode,
    model: int = 3,
    width: int = 100,
    height: int = 100,
    tile_pad: int = 10,
    provider: int = 1,
    device_id: int = 0,
    trt_max_workspace_size: int = 1073741824,
    trt_fp16: bool = False,
    trt_engine_cache: bool = True,
    trt_engine_cache_path: str = dir_name,
    log_level: int = 2,
) -> vs.VideoNode:
    '''
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

    Parameters:
        clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

        model: Model to use.
            0 = RealESRGAN_x2plus (x2 model for general images)
            1 = RealESRGAN_x4plus (x4 model for general images)
            2 = RealESRGAN_x4plus_anime_6B (x4 model optimized for anime images)
            3 = realesr-animevideov3 (x4 model optimized for anime videos)

        tile_w, tile_h: Tile width and height, respectively. As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image. 0 denotes for do not use tile.

        tile_pad: The pad size for each tile, to remove border artifacts.

        provider: The hardware platform to execute on.
            0 = Default CPU
            1 = NVIDIA CUDA (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
            2 = NVIDIA TensorRT (https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)
            3 = DirectML (https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#requirements)
            4 = AMD MIGraphX (https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

        device_id: The device ID.

        trt_max_workspace_size: Maximum workspace size for TensorRT engine.

        trt_fp16: Enable FP16 mode in TensorRT.

        trt_engine_cache: Enable TensorRT engine caching. The purpose of using engine caching is to save engine build time in the case that TensorRT may take
            long time to optimize and build engine. Engine will be cached when it's built for the first time so next time when new inference session is created
            the engine can be loaded directly from cache. In order to validate that the loaded engine is usable for current inference, engine profile is also
            cached and loaded along with engine. If current input shapes are in the range of the engine profile, the loaded engine can be safely used. Otherwise
            if input shapes are out of range, profile cache will be updated to cover the new shape and engine will be recreated based on the new profile (and
            also refreshed in the engine cache). Note each engine is created for specific settings such as model path/name, precision, workspace, profiles etc,
            and specific GPUs and it's not portable, so it's essential to make sure those settings are not changing, otherwise the engine needs to be rebuilt
            and cached again.

            Warning: Please clean up any old engine and profile cache files (.engine and .profile) if any of the following changes:
                Model changes (if there are any changes to the model topology, opset version, operators etc.)
                ORT version changes (i.e. moving from ORT version 1.8 to 1.9)
                TensorRT version changes (i.e. moving from TensorRT 7.0 to 8.0)
                Hardware changes (Engine and profile files are not portable and optimized for specific NVIDIA hardware)

        trt_engine_cache_path: Specify path for TensorRT engine and profile files if trt_engine_cache is true.

        log_level: Log severity level. Applies to session load, initialization, etc.
            0 = Verbose
            1 = Info
            2 = Warning
            3 = Error
            4 = Fatal
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')


        
    model=process_image.get_model("base")
    scale = 2



    modulo = 2 if scale == 2 else 1


    cuda_ep = ('CUDAExecutionProvider', dict(device_id=device_id))

    if provider <= 0:
        providers = ['CPUExecutionProvider']
    elif provider == 1:
        providers = [cuda_ep]
    elif provider == 2:
        providers = [
            (
                'TensorrtExecutionProvider',
                dict(
                    device_id=device_id,
                    trt_max_workspace_size=trt_max_workspace_size,
                    trt_fp16_enable=trt_fp16,
                    trt_engine_cache_enable=trt_engine_cache,
                    trt_engine_cache_path=trt_engine_cache_path,
                ),
            ),
            cuda_ep,
        ]
    elif provider == 3:
        sess_options.enable_mem_pattern = False
        providers = [('DmlExecutionProvider', dict(device_id=device_id))]
    else:
        providers = [('MIGraphXExecutionProvider', dict(device_id=device_id))]


    def realesrgan(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img = frame_to_ndarray(f[0])
        # logging.info('NUMPY')
        # logging.info(str(np.shape(img)))
        # logging.info(str(img))
        img = torch.from_numpy(img[0])
        # logging.info("torch")
        # logging.info(str(img.shape))



        output = process_image.process_frame(model, img, (height, width))
        
        output = torch.unsqueeze(output, 0)
        output = output.cpu().detach().numpy()
        
        # logging.info('Output')
        # logging.info(str(np.shape(output)))
        # logging.info(str(output))
        
        return ndarray_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=width, height=height)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=realesrgan)


def frame_to_ndarray(frame: vs.VideoFrame) -> np.ndarray:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return np.expand_dims(array, axis=0)


def ndarray_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = np.squeeze(array, axis=0)
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame

