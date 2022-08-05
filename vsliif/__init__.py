import math
import os.path as osp

import numpy as np
import onnxruntime as ort
import vapoursynth as vs
import logging
from liif import process_image

import torch

dir_name = osp.dirname(__file__)


def liif_resize(
    clip: vs.VideoNode,
    model: int = 3,
    width: int = 100,
    height: int = 100,
    providers = None,

) -> vs.VideoNode:

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')


        
    model=process_image.get_onnx_model("base",providers=providers)



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


    def liif_resize_frame(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
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
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=liif_resize_frame)


def frame_to_ndarray(frame: vs.VideoFrame) -> np.ndarray:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return np.expand_dims(array, axis=0)


def ndarray_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = np.squeeze(array, axis=0)
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame

