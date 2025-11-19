import abc
import inspect
import argparse
import json
import time

import torch
from PIL import Image, ImageFilter
import torchvision.transforms as T

import ptp_utils
import warnings
from collections import defaultdict
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Union
import math

import torchvision

from diffusers.configuration_utils import FrozenDict
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from models.moodifyclip.load import load, tokenize

warnings.filterwarnings('ignore')
from typing import Union, Tuple, Dict, Optional, List
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
import gradio as gr
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, AutoencoderKL, \
    DiffusionPipeline, StableDiffusionXLPipeline
from diffusers import LCMScheduler

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker, StableDiffusionPipelineOutput

import random
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from models.p2p import seq_aligner
# import seq_aligner
import os
import imageio
import tempfile
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
import copy

from utils import load_512, txt_draw
from models.emotion_stimuli_seg import *
# from models.open_clip_long import factory as open_clip

parser = argparse.ArgumentParser(description='Run the Emotional Image Editing demo app')
parser.add_argument('--use_gradio', action='store_true', default=False, help='Use Gradio interface for demo')
parser.add_argument('--use_clip', default='248', type=str, help='Use CLIP model for text embedding')
parser.add_argument('--use_xl', action='store_true', default=False, help='Use Stable Diffusion XL model')
parser.add_argument('--use_which_clip', default=None, type=str, help='Use which CLIP model')
parser.add_argument('--do_face', action='store_true', default=False, help='Do facial expression editing')
parser.add_argument('--do_mood', action='store_true', default=False, help='Do mood editing')
parser.add_argument('--seed', default=1234, type=int, help='Random seed')
args = parser.parse_args()

if args.do_face:
    NUM_DDIM_STEPS = 40
elif args.do_mood:
    NUM_DDIM_STEPS = 40
LOW_RESOURCE = False

# Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Constants
if args.use_xl:
    MODEL_KEY = "stabilityai/stable-diffusion-xl-base-1.0"
else:
    MODEL_KEY = "CompVis/stable-diffusion-v1-4"
LLAVA_MODEL = "lmms-lab/llama3-llava-next-8b"

CLIP_PATH = "checkpoints/moodifyclip.pt"

DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
FPS = 4
torch_dtype = torch.float32
model_id_or_path = "SimianLuo/LCM_Dreamshaper_v7"

scheduler = LCMScheduler.from_config(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                         subfolder="scheduler")

if args.use_clip == '77':
    MAX_NUM_WORDS = 77
else:
    MAX_NUM_WORDS = 248


def moodifier_sampler(diffusion_scheduler, source_latent, target_latent, step,
                      source_noise, target_noise, original_image, random_noise,
                      stochasticity=1.0, advance_step=True):
    """Custom sampler for inversion-free diffusion editing.

    Args:
        diffusion_scheduler: Scheduler with noise schedule parameters
        source_latent (torch.Tensor): Latent representation of source image
        target_latent (torch.Tensor): Current target latent
        step (int): Current timestep index
        source_noise (torch.Tensor): Predicted noise for source
        target_noise (torch.Tensor): Predicted noise for target
        original_image (torch.Tensor): Original image latent (x0)
        random_noise (torch.Tensor): Random noise for stochastic sampling
        stochasticity (float): Control parameter for noise injection (eta)
        advance_step (bool): Whether to increment scheduler step counter

    Returns:
        Tuple: (updated source latent, updated target latent, predicted x0)
    """

    # Validate scheduler state
    if not diffusion_scheduler.timesteps_initialized:
        raise RuntimeError("Scheduler timesteps must be initialized first")

    # Get current scheduler state
    current_step_idx = diffusion_scheduler.current_step
    next_step_idx = current_step_idx + 1

    # Get timestep values
    current_t = diffusion_scheduler.timesteps[current_step_idx]
    next_t = (diffusion_scheduler.timesteps[next_step_idx]
              if next_step_idx < len(diffusion_scheduler.timesteps)
              else current_t)

    # Get schedule parameters
    alpha_t = diffusion_scheduler.cumulative_alphas[current_t]
    alpha_next = (diffusion_scheduler.cumulative_alphas[next_t]
                  if next_t >= 0 else diffusion_scheduler.final_alpha)

    beta_t = 1 - alpha_t
    beta_next = 1 - alpha_next

    # Calculate noise parameters
    variance = beta_next
    noise_scale = stochasticity * (variance ** 0.5)
    adjusted_noise = noise_scale * random_noise

    # Compute consistent noise term
    consistent_noise = (source_latent - (alpha_t ** 0.5) * original_image) / (1 - alpha_t) ** 0.5

    # Predict denoised image
    pred_original = original_image + (
            (target_latent - source_latent) - (beta_t ** 0.5) * (target_noise - source_noise)
    ) / (alpha_t ** 0.5)

    # Combined noise direction
    noise_direction = (target_noise - source_noise) + consistent_noise
    latent_direction = (beta_next - noise_scale ** 2) ** 0.5 * noise_direction

    # Update latents
    if len(diffusion_scheduler.timesteps) > 1:
        new_target = (alpha_next ** 0.5) * pred_original + latent_direction + adjusted_noise
        new_source = (alpha_next ** 0.5) * original_image + latent_direction + adjusted_noise
    else:
        new_target = pred_original
        new_source = original_image

    # Advance scheduler if requested
    if advance_step:
        diffusion_scheduler.advance_step()

    return new_source, new_target, pred_original

class EditPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: LCMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)


        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker
        )
        self.unet = unet
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            # text_inputs = self.tokenizer(
            #     prompt,
            #     padding="max_length",
            #     max_length=self.tokenizer.model_max_length,
            #     truncation=True,
            #     return_tensors="pt",
            # )
            # text_input_ids = text_inputs.input_ids
            text_input_ids = self.tokenizer(prompt)
            # untruncated_ids = self.tokenizer(prompt,
            #                                  padding="longest",
            #                                  return_tensors="pt").input_ids
            untruncated_ids = self.tokenizer(prompt)

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.decoder(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            # if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            #     # attention_mask = text_inputs.attention_mask.to(device)
            #     attention_mask = text_input_ids.attention_mask.to(device)
            # else:
            #     attention_mask = None

            # prompt_embeds = self.text_encoder(
            #     text_input_ids.to(device),
            #     attention_mask=attention_mask,
            # )
            # prompt_embeds = prompt_embeds[0]
            prompt_embeds = self.text_encoder(text_input_ids.to(device))

        if self.text_encoder is not None:
            # prompt_embeds_dtype = self.text_encoder.dtype
            prompt_embeds_dtype = torch.float16
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            # uncond_input = self.tokenizer(
            #     uncond_tokens,
            #     padding="max_length",
            #     max_length=max_length,
            #     truncation=True,
            #     return_tensors="pt",
            # )
            uncond_input = self.tokenizer(negative_prompt)

            # if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            #     attention_mask = uncond_input.attention_mask.to(device)
            # else:
            #     attention_mask = None

            # negative_prompt_embeds = self.text_encoder(
            #     uncond_input.input_ids.to(device),
            #     attention_mask=attention_mask,
            # )
            negative_prompt_embeds = self.text_encoder(uncond_input.to(device))
            negative_prompt_embeds = negative_prompt_embeds[0]
            # print('negative_prompt_embeds.shape 1: ', negative_prompt_embeds.shape)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # print('negative_prompt_embeds.shape 2: ', negative_prompt_embeds.shape)

            # negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # print('prompt_embeds.shape: ', prompt_embeds.shape)
        # print('negative_prompt_embeds.shape: ', negative_prompt_embeds.shape)
        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.check_inputs
    def check_inputs(
        self, prompt, strength, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, denoise_model, generator=None):
        image = image.to(device=device, dtype=dtype)

        batch_size = image.shape[0]

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                image = image.float()
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        # add noise to latents using the timestep
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        clean_latents = init_latents
        if denoise_model:
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents
        else:
            latents = noise

        return latents, clean_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        source_prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]]=None,
        positive_prompt: Union[str, List[str]]=None,
        image: PipelineImageInput = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        original_inference_steps: Optional[int]  = 50,
        guidance_scale: Optional[float] = 7.5,
        source_guidance_scale: Optional[float] = 1,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        denoise_model: Optional[bool] = True,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds_tuple = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        source_prompt_embeds_tuple = self.encode_prompt(
            source_prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            positive_prompt, None
        )
        if prompt_embeds_tuple[1] is not None:
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
        else:
            prompt_embeds = prompt_embeds_tuple[0]
        if source_prompt_embeds_tuple[1] is not None:
            source_prompt_embeds = torch.cat([source_prompt_embeds_tuple[1], source_prompt_embeds_tuple[0]])
        else:
            source_prompt_embeds = source_prompt_embeds_tuple[0]

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(
          num_inference_steps=num_inference_steps,
          device=device,
          original_inference_steps=original_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents, clean_latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, denoise_model, generator
        )
        source_latents = latents
        mutual_latents = latents

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        generator = extra_step_kwargs.pop("generator", None)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            edited_latents = []
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                source_latent_model_input = (
                    torch.cat([source_latents] * 2) if do_classifier_free_guidance else source_latents
                )
                mutual_latent_model_input = (
                    torch.cat([mutual_latents] * 2) if do_classifier_free_guidance else mutual_latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                source_latent_model_input = self.scheduler.scale_model_input(source_latent_model_input, t)
                mutual_latent_model_input = self.scheduler.scale_model_input(mutual_latent_model_input, t)

                # predict the noise residual
                if do_classifier_free_guidance:
                    concat_latent_model_input = torch.stack(
                        [
                            source_latent_model_input[0],
                            latent_model_input[0],
                            mutual_latent_model_input[0],
                            source_latent_model_input[1],
                            latent_model_input[1],
                            mutual_latent_model_input[1],
                        ],
                        dim=0,
                    )
                    concat_prompt_embeds = torch.stack(
                        [
                            source_prompt_embeds[0],
                            prompt_embeds[0],
                            source_prompt_embeds[0],
                            source_prompt_embeds[1],
                            prompt_embeds[1],
                            source_prompt_embeds[1],
                        ],
                        dim=0,
                    )
                else:
                    concat_latent_model_input = torch.cat(
                        [
                            source_latent_model_input,
                            latent_model_input,
                            mutual_latent_model_input,
                        ],
                        dim=0,
                    )
                    concat_prompt_embeds = torch.cat(
                        [
                            source_prompt_embeds,
                            prompt_embeds,
                            source_prompt_embeds,
                        ],
                        dim=0,
                    )

                concat_noise_pred = self.unet(
                    concat_latent_model_input,
                    t,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_hidden_states=concat_prompt_embeds,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    (
                        source_noise_pred_uncond,
                        noise_pred_uncond,
                        mutual_noise_pred_uncond,
                        source_noise_pred_text,
                        noise_pred_text,
                        mutual_noise_pred_text
                    ) = concat_noise_pred.chunk(6, dim=0)

                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    source_noise_pred = source_noise_pred_uncond + source_guidance_scale * (
                        source_noise_pred_text - source_noise_pred_uncond
                    )
                    mutual_noise_pred = mutual_noise_pred_uncond + source_guidance_scale * (
                        mutual_noise_pred_text - mutual_noise_pred_uncond
                    )

                else:
                    (source_noise_pred, noise_pred, mutual_noise_pred) = concat_noise_pred.chunk(3, dim=0)

                noise = torch.randn(
                    latents.shape, dtype=latents.dtype, device=latents.device, generator=generator
                )

                _, latents, pred_x0 = moodifier_sampler(
                  self.scheduler, source_latents,
                  latents, t,
                  source_noise_pred, noise_pred,
                  clean_latents, noise=noise,
                  eta=eta, to_next=False,
                  **extra_step_kwargs
                )

                edited_latents.append(pred_x0)

                source_latents, mutual_latents, pred_xm = moodifier_sampler(
                  self.scheduler, source_latents,
                  mutual_latents, t,
                  source_noise_pred, mutual_noise_pred,
                  clean_latents, noise=noise,
                  eta=eta, **extra_step_kwargs
                )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        mutual_latents, latents = callback(i, t, source_latents, latents, mutual_latents, alpha_prod_t)

        # 9. Post-processing
        if not output_type == "latent":
            edited_images = []
            has_nsfw_concepts = []
            for i in range(len(edited_latents)):
                image = self.vae.decode(edited_latents[i] / self.vae.config.scaling_factor, return_dict=False)[0]
                # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
                edited_images.append(image)
                # has_nsfw_concepts.append(has_nsfw_concept)
            # image = self.vae.decode(pred_x0 / self.vae.config.scaling_factor, return_dict=False)[0]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)


        has_nsfw_concept = None

        post_processed_images = []
        for i in range(len(edited_images)):
            # has_nsfw_concept = has_nsfw_concepts[i]
            # if has_nsfw_concept is None:
            #     do_denormalize = [True] * image.shape[0]
            # else:
            #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(edited_images[i], output_type=output_type)
            post_processed_images.append(image)

        if not return_dict:
            return (post_processed_images, has_nsfw_concepts)

        return StableDiffusionPipelineOutput(images=post_processed_images, nsfw_content_detected=has_nsfw_concepts)

print(f'[INFO] loading CLIP from {CLIP_PATH}...')
vitl_model, vitl_preprocess = load(CLIP_PATH,
                                       device=DEVICE)

vitl_model.eval()
vitL_encoder = vitl_model.encode_text_full

pipe = EditPipeline.from_pretrained(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                        scheduler=scheduler, torch_dtype=torch_dtype)

encoder = vitL_encoder
tokenizer = tokenize
pipe.text_encoder = vitL_encoder
pipe.tokenizer = tokenize


if torch.cuda.is_available():
    pipe = pipe.to(DEVICE)

class ModelLoader:
    def __init__(self):
        self.llava_tokenizer = None
        self.llava_model = None
        self.image_processor = None
        self.load_models()

    def load_models(self):
        # Load LLaVA model
        self.llava_tokenizer, self.llava_model, self.image_processor, _ = load_pretrained_model(
            LLAVA_MODEL,
            None,
            "llava_llama_3"
        )
        self.llava_model.eval()
        self.llava_model.tie_weights()


model_loader = ModelLoader()


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


def save_images(images, dest, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    pil_img = Image.fromarray(images[-1])
    pil_img.save(dest)
    print(f'[INFO] saved image to {dest}!!')

    # display(pil_img)


def gaussian_blur(tensor: torch.Tensor,
                  kernel_size: int,
                  sigma: float) -> torch.Tensor:
    channels = tensor.dim()
    if channels == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif channels == 3:
        tensor = tensor.unsqueeze(0)

    padding = (kernel_size - 1) // 2
    blur = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    blurred = blur(tensor)

    if channels == 2:
        blurred = blurred.squeeze(0).squeeze(0)
    elif channels == 3:
        blurred = blurred.squeeze(0)

    return blurred

def load_binary_mask(mask_input, target_size=(612, 408)):
    if isinstance(mask_input, str):
        mask = Image.open(mask_input).convert('L')
    elif isinstance(mask_input, np.ndarray):
        mask = Image.fromarray((mask_input * 255).astype(np.uint8))
    elif isinstance(mask_input, torch.Tensor):
        mask = Image.fromarray((mask_input.cpu().numpy() * 255).astype(np.uint8))
    else:
        raise TypeError("Unsupported mask input type")

    # transform = T.Compose([T.ToTensor()])
    # mask_tensor = transform(mask).float()
    # if mask_tensor.dim() == 3:
    #     mask_tensor = mask_tensor.unsqueeze(0)
    # return mask_tensor

    # Add Gaussian smoothing before tensor conversion
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))

    transform = T.Compose([
        T.ToTensor(),
        T.GaussianBlur(kernel_size=3, sigma=1.0)  # Additional smoothing
    ])
    mask_tensor = transform(mask).float()
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
    return mask_tensor


class AttentionModulator:
    def __init__(self,
                 min_threshold: float = 3.0,
                 max_threshold: float = 7.0,
                 preserve_ratio: float = 0.3,
                 spatial_decay: float = 2.0,
                 blend_sharpness: float = 2.0,
                 boundary_smooth_ratio: float = 0.05):

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.preserve_ratio = preserve_ratio
        self.spatial_decay = spatial_decay
        self.blend_sharpness = blend_sharpness
        self.boundary_smooth_ratio = boundary_smooth_ratio

    def compute_adaptive_importance_weight(self,
                                           importance_score: Union[float, torch.Tensor]) -> torch.Tensor:

        if isinstance(importance_score, (int, float)):
            importance_score = torch.tensor(importance_score)

        # Center score around midpoint of thresholds
        centered_score = importance_score - (self.min_threshold + self.max_threshold) / 2

        # Compute base weight with softer boundaries
        base_weight = torch.sigmoid(centered_score * 0.5)

        # Apply gaussian decay for extreme values
        mean = (self.min_threshold + self.max_threshold) / 2
        decay = torch.exp(-0.5 * torch.pow((importance_score - mean) / 2, 2))

        return base_weight * decay

    def compute_spatial_weights(self,
                                latent_binary: torch.Tensor) -> torch.Tensor:
        # print('latent_binary: ', latent_binary.shape)
        # Get coordinates as a tensor of [N, 2] shape
        coords = torch.nonzero(latent_binary)
        if len(coords) == 0:  # If mask is empty
            return torch.zeros_like(latent_binary)

        # print('coords: ', coords.shape)
        # Calculate center of mass from all coordinates
        center_y = coords[:, 0].float().mean()
        center_x = coords[:, 1].float().mean()

        # Initialize weights tensor
        weights = torch.zeros_like(latent_binary, dtype=torch.float32)

        # Calculate max possible distance for normalization
        max_dist = torch.sqrt(torch.tensor(latent_binary.shape[0] ** 2 +
                                           latent_binary.shape[1] ** 2))

        # Simply loop through all coordinates
        for coord in coords:
            y, x = coord[0], coord[1]
            dist = torch.sqrt(torch.pow(y.float() - center_y, 2) +
                              torch.pow(x.float() - center_x, 2))
            weights[y, x] = torch.exp(-dist / max_dist * self.spatial_decay)

        return weights

    def apply_progressive_scaling(self,
                                  masked_attention: torch.Tensor,
                                  importance_weight: torch.Tensor) -> torch.Tensor:

        # Calculate statistics
        valid_attention = masked_attention[masked_attention != 0]
        if len(valid_attention) == 0:
            return masked_attention

        mask_mean = valid_attention.mean()
        mask_std = valid_attention.std()

        if mask_std == 0:
            return masked_attention

        # Compute deviation-based scaling
        deviation = (masked_attention - mask_mean).abs() / mask_std
        scaling_multiplier = torch.exp(-deviation * self.preserve_ratio)

        # Apply scaled modification
        base_scaling = 1 + (importance_weight - 0.5)
        adaptive_scaling = base_scaling * scaling_multiplier
        # print('adaptive_scaling: ', adaptive_scaling)
        # adaptive_scaling = 1
        return mask_mean + (masked_attention - mask_mean) * adaptive_scaling

    def apply_localized_modulation(self,
                                   scaled_attention: torch.Tensor,
                                   latent_binary: torch.Tensor,
                                   spatial_weights: torch.Tensor,
                                   importance_weight: torch.Tensor) -> torch.Tensor:

        local_scaling = 1 + (importance_weight - 0.5) * spatial_weights
        modulated = scaled_attention * local_scaling

        return torch.where(latent_binary > 0, modulated, scaled_attention)

    def modulate_attention(self,
                           latent_attention: torch.Tensor,
                           latent_binary: torch.Tensor,
                           importance_score: Optional[float] = None) -> torch.Tensor:

        if importance_score is None:
            return latent_attention

        # Compute importance weight
        importance_weight = self.compute_adaptive_importance_weight(importance_score)

        # Get masked attention region
        masked_attention = latent_attention * latent_binary

        # Compute spatial weights
        spatial_weights = self.compute_spatial_weights(latent_binary)

        # Apply progressive scaling
        scaled_attention = self.apply_progressive_scaling(masked_attention, importance_weight)

        # Apply localized modulation
        modulated_attention = self.apply_localized_modulation(
            scaled_attention, latent_binary, spatial_weights, importance_weight)

        # Smooth blending
        blend_factor = torch.sigmoid(importance_weight * self.blend_sharpness - 1)
        final_attention = (1 - blend_factor) * latent_attention + \
                          blend_factor * modulated_attention

        # Boundary smoothing
        kernel_size = max(3, int(latent_attention.shape[0] * self.boundary_smooth_ratio))
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed_attention = gaussian_blur(
            final_attention, kernel_size, sigma=kernel_size / 6)

        return smoothed_attention


# Example usage:
def apply_attention_modulation(latent_attention: torch.Tensor,
                               latent_binary: torch.Tensor,
                               importance_score: float) -> torch.Tensor:
    modulator = AttentionModulator(
        min_threshold=0.0,
        max_threshold=3.0,
        preserve_ratio=0.1,
        spatial_decay=2.0,
        blend_sharpness=2.0,
        boundary_smooth_ratio=0.05
    )

    return modulator.modulate_attention(latent_attention, latent_binary, importance_score)


class LocalBlend:

    def get_mask(self, x_t, maps, word_idx, thresh, i):
        maps = maps * word_idx.reshape(1, 1, 1, 1, -1)
        maps = (maps[:, :, :, :, 1:self.len - 1]).mean(0, keepdim=True)
        maps = (maps).max(-1)[0]
        maps = torch.nn.functional.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        if args.do_face:
            mask = maps > thresh
        else:
            mask = maps > torch.quantile(maps, 0.75)
        mask = maps
        return mask

    def save_image(self, mask, i, caption):
        image = mask[0, 0, :, :]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if not os.path.exists(f"output1/{caption}"):
            os.mkdir(f"output1/{caption}")
        save_images(image, f"output1/{caption}/{i}.jpg")

    def refine_attn_with_emo_stimulus(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15,
                                      use_xm=False):
        if args.use_xl:
            maps = attention_store["down_cross"][20:40] + attention_store["up_cross"][:30]
        else:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]

        h, w = x_t.shape[2], x_t.shape[3]
        h, w = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
        maps = [
            item.reshape(2, -1, 1, h // int((h * w / item.shape[-2]) ** 0.5), w // int((h * w / item.shape[-2]) ** 0.5),
                         MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps_s = maps[0, :]
        maps_m = maps[1, :]

        thresh_e = temperature / alpha_prod ** (0.5)
        if thresh_e < self.thresh_e:
            thresh_e = self.thresh_e
        thresh_m = self.thresh_m

        def load_binary_mask(mask_input, target_size=(612, 408)):
            if isinstance(mask_input, str):
                mask = Image.open(mask_input).convert('L')
            elif isinstance(mask_input, np.ndarray):
                mask = Image.fromarray((mask_input * 255).astype(np.uint8))
            elif isinstance(mask_input, torch.Tensor):
                mask = Image.fromarray((mask_input.cpu().numpy() * 255).astype(np.uint8))
            else:
                raise TypeError("Unsupported mask input type")

            # transform = T.Compose([T.ToTensor()])
            # mask_tensor = transform(mask).float()
            # if mask_tensor.dim() == 3:
            #     mask_tensor = mask_tensor.unsqueeze(0)
            # return mask_tensor

            # Add Gaussian smoothing before tensor conversion
            mask = mask.filter(ImageFilter.GaussianBlur(radius=2))

            transform = T.Compose([
                T.ToTensor(),
                T.GaussianBlur(kernel_size=3, sigma=1.0)  # Additional smoothing
            ])
            mask_tensor = transform(mask).float()
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(0)
            return mask_tensor

        def combine_attention_masks(binary_mask, latent_attention, importance_score=None, targeted_stimuli=None,
                                    threshold=0.9):
            # Add cache as a function attribute if it doesn't exist
            if not hasattr(self, '_cached_attention'):
                self._cached_attention = {}

            # Create a unique key for the cache based on binary_mask shape and targeted_stimuli
            cache_key = f"{targeted_stimuli}_{binary_mask.shape}"

            # # If we have a cached mask for this stimuli, return it
            # if cache_key in self._cached_attention:
            #     return self._cached_attention[cache_key]

            # First time processing - do the full computation
            latent_binary = F.interpolate(
                binary_mask.float(),
                size=(latent_attention.shape[-2], latent_attention.shape[-1]),
                mode='bilinear',
                align_corners=False
            )
            latent_binary = latent_binary.expand(-1, 4, -1, -1)

            feature_patterns = {
                'small': {
                    'strength': 2.0,
                    'falloff': 0.8,
                    'type': 'focal'
                },
                'medium': {
                    'strength': 1.5,
                    'falloff': 0.85,
                    'type': 'balanced'
                },
                'large': {
                    'strength': 1.2,
                    'falloff': 0.95,
                    'type': 'region'
                }
            }

            def get_object_pattern(binary_mask):
                total_pixels = binary_mask.shape[-2] * binary_mask.shape[-1]
                object_pixels = torch.sum(binary_mask > 0.5)
                size_ratio = object_pixels / total_pixels
                # print('size_ratio: ', size_ratio)
                if size_ratio < 0.15:
                    return feature_patterns['small']
                elif size_ratio < 0.4:
                    return feature_patterns['medium']
                else:
                    return feature_patterns['large']

            def create_center_weighted_mask(mask, sharpness=0.7):
                center_y, center_x = torch.where(mask[0, 0] > 0.5)
                if len(center_y) > 0 and len(center_x) > 0:
                    center_y = center_y.float().mean()
                    center_x = center_x.float().mean()
                    y, x = torch.meshgrid(
                        torch.arange(mask.shape[2], device=mask.device),
                        torch.arange(mask.shape[3], device=mask.device)
                    )
                    distance = torch.sqrt((y.float() - center_y) ** 2 + (x.float() - center_x) ** 2)
                    max_dist = torch.sqrt(torch.tensor(mask.shape[2] ** 2 + mask.shape[3] ** 2))
                    falloff = torch.exp(-(distance / max_dist) / sharpness)
                    return falloff.unsqueeze(0).unsqueeze(0)
                return torch.ones_like(mask)

            def get_enhancement_type(binary_mask):
                pattern = get_object_pattern(binary_mask)
                if pattern['type'] == 'focal':
                    return create_center_weighted_mask(binary_mask, sharpness=pattern['falloff'])
                elif pattern['type'] == 'balanced':
                    center_mask = create_center_weighted_mask(binary_mask, sharpness=pattern['falloff'])
                    region_mask = torch.ones_like(binary_mask)
                    return (center_mask + region_mask) / 2
                else:  # region
                    base_mask = torch.ones_like(binary_mask)
                    edge_mask = F.max_pool2d(binary_mask, 3, stride=1, padding=1) - binary_mask
                    return base_mask + edge_mask * 0.3

            # Apply size-adaptive enhancement
            params = get_object_pattern(latent_binary)
            enhancement_mask = get_enhancement_type(latent_binary)

            # Scale enhancement based on importance score
            if importance_score is not None:
                scale = torch.sigmoid(torch.tensor(importance_score - 5.0)) * 0.5 + 0.75
                params['strength'] = params['strength'] * scale

            image_data = latent_binary[0, 0].cpu().numpy()
            image_data = ((image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())).astype(
                np.uint8)
            # save_images(image_data,
            #             f"output1/masks/latent_binary_initial_{i}_{targeted_stimuli}_{importance_score}.jpg")

            # Apply enhancement with dynamic strength
            enhancement = 1.0 + (params['strength'] - 1.0) * enhancement_mask
            latent_binary = latent_binary * enhancement

            # Normalize masks
            latent_binary = latent_binary / (latent_binary.max() + 1e-8)

            latent_attention = latent_attention / (latent_attention.max() + 1e-8)
            latent_attention = latent_attention.to(latent_binary.device)

            image_data = latent_binary[0, 0].cpu().numpy()
            image_data = ((image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())).astype(
                np.uint8)
            # save_images(image_data,
            #             f"output1/masks/latent_binary_beforeattnmodule_{i}_{targeted_stimuli}_{importance_score}.jpg")

            image_data = latent_attention[0, 0].cpu().numpy()
            image_data = ((image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())).astype(
                np.uint8)
            # save_images(image_data,
            #             f"output1/masks/latent_attention_initial_{i}_{targeted_stimuli}_{importance_score}.jpg")

            # Apply importance-based modulation
            if importance_score is not None:
                latent_attention = apply_attention_modulation(
                    latent_attention, latent_binary, importance_score)

            image_data = latent_binary[0, 0].cpu().numpy()
            image_data = ((image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())).astype(
                np.uint8)
            # save_images(image_data,
            #             f"output1/masks/latent_binary_afterapplyattn_{i}_{targeted_stimuli}_{importance_score}.jpg")

            image_data = latent_attention[0, 0].cpu().numpy()
            image_data = ((image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())).astype(
                np.uint8)
            # save_images(image_data,
            #             f"output1/masks/latent_attention_afterapplyattn_{i}_{targeted_stimuli}_{importance_score}.jpg")

            # Combine masks with dynamic blending
            combined = latent_binary * torch.pow(latent_attention, 0.8)

            image_data = combined[0, 0].cpu().numpy()
            image_data = ((image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())).astype(
                np.uint8)
            # save_images(image_data,
            #             f"output1/masks/combined_{i}_{targeted_stimuli}_{importance_score}.jpg")

            # Dynamic thresholding based on pattern type
            params = get_object_pattern(latent_binary)
            if params['type'] == 'focal':
                high_thresh, low_thresh = 0.6, 0.3
            elif params['type'] == 'balanced':
                high_thresh, low_thresh = 0.4, 0.2
            else:  # region
                high_thresh, low_thresh = 0.2, 0.1

            high_mask = combined > high_thresh
            low_mask = combined > low_thresh

            # Create smooth transition
            smooth_mask = F.avg_pool2d(high_mask.float(), 3, stride=1, padding=1)
            transition = torch.where(
                (smooth_mask > 0) & (smooth_mask < 1),
                combined * (smooth_mask * 0.6 + 0.3),
                combined
            )

            refined_attention = torch.where(
                high_mask | low_mask,
                transition,
                torch.zeros_like(combined)
            )

            # Final normalization with pattern-specific contrast
            refined_attention = refined_attention / (
                    refined_attention.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)

            refined_attention = torch.pow(refined_attention, params['falloff'])

            # Save visualization
            image_data = refined_attention[0, 0].cpu().numpy()
            image_data = ((image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())).astype(
                np.uint8)
            # save_images(image_data, f"output1/masks/refined_attention_{i}_{targeted_stimuli}_{importance_score}.jpg")

            # Cache the computed attention mask
            self._cached_attention[cache_key] = refined_attention

            return refined_attention

        def load_and_process_masks(mask_input, attn_map, importance_score=None, targeted_stimuli=None):
            binary_mask = load_binary_mask(mask_input)
            refined_attention = combine_attention_masks(
                binary_mask=binary_mask,
                latent_attention=attn_map,
                importance_score=importance_score,
                targeted_stimuli=targeted_stimuli,
            )
            return refined_attention

        mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
        mask_m = self.get_mask(x_t, maps_s, (self.alpha_m - self.alpha_me), thresh_m, i)
        mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)

        general_global_attn_e = self.save_image(mask_e, i, "mask_e")

        # Define timesteps for nuanced attention application
        REFINED_STEPS = {
            'early': list(range(0, 8)),  # Strong initial structure
            'early_mid': list(range(8, 16)),  # Detail refinement
            'middle': list(range(16, 24)),  # Balance features
            'late_mid': list(range(24, 32)),  # Maintain structure
            'late': list(range(32, 40))  # Final touches
        }

        # Stage-specific parameters
        STAGE_PARAMS = {
            'early': {'base_strength': 2.0, 'context_preserve': 0.7},
            'early_mid': {'base_strength': 1.8, 'context_preserve': 0.75},
            'middle': {'base_strength': 1.5, 'context_preserve': 0.8},
            'late_mid': {'base_strength': 1.3, 'context_preserve': 0.85},
            'late': {'base_strength': 1.2, 'context_preserve': 0.9}
        }

        if len(self.emotion_stimuli_masks) != 0 and np.mean(self.importance_scores) != 10:
            is_refinement_step = any(i in steps for steps in REFINED_STEPS.values())

            if is_refinement_step:
                print(f'Applying refined attention at step {i}')
                combined_refined_attn_mask = torch.zeros_like(mask_e, dtype=torch.float32)
                general_global_combined_refined_attn_mask = torch.zeros_like(mask_e, dtype=torch.float32)

                stage = next(name for name, steps in REFINED_STEPS.items() if i in steps)
                stage_params = STAGE_PARAMS[stage]

                print(f'Current stage: {stage}, params: {stage_params}')

                for emotion_stimuli_mask, importance, targeted_stimuli in (
                        zip(self.emotion_stimuli_masks,
                            self.importance_scores,
                            self.emotion_stimuli_objs)):

                    adjusted_importance = importance * stage_params['base_strength']

                    if targeted_stimuli == 'background':
                        general_global_refined_attn = load_and_process_masks(
                            emotion_stimuli_mask,
                            attn_map=mask_e,
                            importance_score=adjusted_importance * stage_params['context_preserve'],
                            targeted_stimuli=targeted_stimuli
                        )
                        general_global_refined_attn = general_global_refined_attn.to(x_t.device)
                        general_global_combined_refined_attn_mask += \
                            general_global_refined_attn.max(dim=1, keepdim=True)[0]
                    else:
                        refined_attn = load_and_process_masks(
                            emotion_stimuli_mask,
                            attn_map=mask_e,
                            importance_score=adjusted_importance,
                            targeted_stimuli=targeted_stimuli
                        )
                        refined_attn = refined_attn.to(x_t.device)
                        combined_refined_attn_mask += refined_attn.max(dim=1, keepdim=True)[0]

                print(f'Applied attention refinement at step {i} with stage params {stage_params}')

                # mask_e = combined_refined_attn_mask * stage_params['base_strength']
                mask_e = combined_refined_attn_mask
                general_global_mask_e = general_global_combined_refined_attn_mask * stage_params['context_preserve']
            else:
                print(f'Using original attention at step {i}')
                mask_e = mask_e
                general_global_mask_e = mask_e
        else:
            print('no emotion stimuli masks')

        attn_e = self.save_image(mask_e, i, "mask_e")
        self.save_image(mask_m, i, "mask_m")
        self.save_image(mask_me, i, "mask_me")

        print('self.alpha_e.sum(): ', self.alpha_e.sum())

        if self.alpha_e.sum() == 0:
            x_t_out = x_t
            general_global_x_t_out = x_t
        else:
            print('here!!')
            x_t_out = torch.where(mask_e > 0, x_t, x_m)
            # general_global_x_t_out = torch.where(general_global_mask_e > 0, x_t, x_m)


        x_t_out = torch.where(mask_m > 0, x_s, x_t_out)
        # general_global_x_t_out = torch.where(mask_m > 0, x_s, general_global_x_t_out)

        return x_m, x_t_out

    def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False):
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        h, w = x_t.shape[2], x_t.shape[3]
        h, w = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
        maps = [
            item.reshape(2, -1, 1, h // int((h * w / item.shape[-2]) ** 0.5), w // int((h * w / item.shape[-2]) ** 0.5),
                         MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps_s = maps[0, :]
        maps_m = maps[1, :]
        thresh_e = temperature / alpha_prod ** (0.5)
        if thresh_e < self.thresh_e:
            thresh_e = self.thresh_e
        thresh_m = self.thresh_m
        mask_e, map_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
        mask_m, map_m = self.get_mask(x_t, maps_s, (self.alpha_m - self.alpha_me), thresh_m, i)
        mask_me, map_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)
        if self.save_inter:
            self.save_image(mask_e, i, "mask_e")
            self.save_image(mask_m, i, "mask_m")
            self.save_image(mask_me, i, "mask_me")

        combined_mask_e = torch.zeros_like(mask_e)
        for emotion_mask, importance_score, target in zip(self.emotion_stimuli_masks,
                                                          self.importance_scores,
                                                          self.emotion_stimuli_objs):
            emotion_mask = load_binary_mask(emotion_mask)
            emotion_mask = F.interpolate(emotion_mask.float(),
                                         size=mask_e.shape[-2:],
                                         mode='bilinear',
                                         align_corners=False)
            emotion_mask = emotion_mask.to(mask_e.device)
            scale_factor = 0.5 + torch.sigmoid(torch.tensor(importance_score - 5.0))
            print('scale_factor: ', scale_factor)
            if scale_factor > 1:
                combined_mask_e = torch.where(emotion_mask>0, mask_e, combined_mask_e)
            # else:
                  # add it back for nature?
            #     combined_mask_e = torch.where(emotion_mask>0, map_e > 0.1, combined_mask_e)

            # enhanced_mask = constrained_mask * scale_factor
            # combined_mask_e = combined_mask_e + enhanced_mask
            # combined_mask_e = combined_mask_e + constrained_mask
        # combined_mask_e = combined_mask_e / (combined_mask_e.max() + 1e-8)
        self.save_image(combined_mask_e, i, "combined_mask_e")

        if self.alpha_e.sum() == 0:
            x_t_out = x_t
        else:
            print('combined_mask_e: ', combined_mask_e.shape, combined_mask_e.max(), combined_mask_e.min())
            print('mask_m: ', mask_m.shape, mask_m.max(), mask_m.min())
            # x_t_out = x_m + combined_mask_e * (x_t - x_m)
            combined_mask_e_bool = combined_mask_e > 0
            x_t_out = torch.where(combined_mask_e_bool, x_t, x_m)

        x_t_out = torch.where(mask_m, x_s, x_t_out)
        if use_xm:
            x_t_out = torch.where(mask_me, x_m, x_t_out)

        # if self.alpha_e.sum() == 0:
        #     x_t_out = x_t
        # else:
        #     print('mask_m: ', mask_m.shape, mask_m.max(), mask_m.min())
        #     x_t_out = torch.where(mask_e, x_t, x_m)
        # x_t_out = torch.where(mask_m, x_s, x_t_out)
        # if use_xm:
        #     x_t_out = torch.where(mask_me, x_m, x_t_out)

        return x_m, x_t_out

    def __init__(self, thresh_e=0.3, thresh_m=0.3, emotion_stimuli_objs=None, emotion_stimuli_masks=None,
                 importance_scores=None, save_inter=False):
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m
        self.save_inter = save_inter
        self.emotion_stimuli_objs = emotion_stimuli_objs
        self.emotion_stimuli_masks = emotion_stimuli_masks
        self.importance_scores = importance_scores

    def set_map(self, ms, alpha, alpha_e, alpha_m, len):
        self.m = ms
        self.alpha = alpha
        self.alpha_e = alpha_e
        self.alpha_m = alpha_m
        alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
        self.alpha_me = alpha_me.to(torch.float)
        self.len = len


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def self_attn_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, i, t, x_s, x_t, x_m, alpha_prod):
        if (self.local_blend is not None) and (i > 0):
            use_xm = (self.cur_step + self.start_steps + 1 == self.num_steps)
            # x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
            x_m, x_t = self.local_blend.refine_attn_with_emo_stimulus(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
        return x_m, x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out

    def self_attn_forward(self, q, k, v, num_heads):
        if q.shape[0] // num_heads == 3:
            if (self.self_replace_steps <= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                q = torch.cat([q[:num_heads * 2], q[num_heads:num_heads * 2]])
                k = torch.cat([k[:num_heads * 2], k[:num_heads]])
                v = torch.cat([v[:num_heads * 2], v[:num_heads]])
            else:
                q = torch.cat([q[:num_heads], q[:num_heads], q[:num_heads]])
                k = torch.cat([k[:num_heads], k[:num_heads], k[:num_heads]])
                v = torch.cat([v[:num_heads * 2], v[:num_heads]])
            return q, k, v
        else:
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            if (self.self_replace_steps <= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                qu = torch.cat([qu[:num_heads * 2], qu[num_heads:num_heads * 2]])
                qc = torch.cat([qc[:num_heads * 2], qc[num_heads:num_heads * 2]])
                ku = torch.cat([ku[:num_heads * 2], ku[:num_heads]])
                kc = torch.cat([kc[:num_heads * 2], kc[:num_heads]])
                vu = torch.cat([vu[:num_heads * 2], vu[:num_heads]])
                vc = torch.cat([vc[:num_heads * 2], vc[:num_heads]])
            else:
                qu = torch.cat([qu[:num_heads], qu[:num_heads], qu[:num_heads]])
                qc = torch.cat([qc[:num_heads], qc[:num_heads], qc[:num_heads]])
                ku = torch.cat([ku[:num_heads], ku[:num_heads], ku[:num_heads]])
                kc = torch.cat([kc[:num_heads], kc[:num_heads], kc[:num_heads]])
                vu = torch.cat([vu[:num_heads * 2], vu[:num_heads]])
                vc = torch.cat([vc[:num_heads * 2], vc[:num_heads]])

            return torch.cat([qu, qc], dim=0), torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce, attn_masa = attn[0], attn[1], attn[2]
            attn_replace_new = self.replace_cross_attention(attn_masa, attn_repalce)
            attn_base_store = self.replace_cross_attention(attn_base, attn_repalce)
            if (self.cross_replace_steps >= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                attn[1] = attn_base_store
            attn_store = torch.cat([attn_base_store, attn_replace_new])
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            attn_store = attn_store.reshape(2 * h, *attn_store.shape[2:])
            super(AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
        return attn

    def __init__(self, prompts, num_steps: int, start_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts) + 1
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device).to(torch_dtype)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_masa, att_replace):
        attn_masa_replace = attn_masa[:, :, self.mapper].squeeze()
        attn_replace = attn_masa_replace * self.alphas + \
                       att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, prompt_specifiers, num_steps: int, start_steps: int, cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, start_steps, cross_replace_steps, self_replace_steps,
                                              local_blend)
        device = DEVICE
        self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers,
                                                                                      tokenizer, encoder, device,
                                                                                      max_len=MAX_NUM_WORDS)
        self.mapper, alphas, ms = self.mapper.to(device), alphas.to(device).to(torch_dtype), ms.to(device).to(
            torch_dtype)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
        ms = ms.to(device)
        alpha_e = alpha_e.to(device)
        alpha_m = alpha_m.to(device)
        # t_len = len(tokenizer(prompts[1])["input_ids"])
        try:
            tgt_prompt = [prompts[1]]
            t_len = len(tokenizer(tgt_prompt)[0])
        except:
            tgt_prompt = prompts[1]
            t_len = len(tokenizer(tgt_prompt,
                                  padding="max_length",
                                  max_length=77,
                                  truncation=True,
                                  return_tensors="pt").input_ids[0])
        self.local_blend.set_map(ms, alphas, alpha_e, alpha_m, t_len)


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), MAX_NUM_WORDS)
    values = torch.tensor(values, dtype=torch_dtype)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def inference(img, source_prompt, target_prompt,
              local, mutual,
              positive_prompt, negative_prompt,
              guidance_s, guidance_t,
              num_inference_steps,
              width, height, seed, strength,
              cross_replace_steps, self_replace_steps,
              thresh_e, thresh_m, denoise,
                emotion_stimulus_lst, emotion_stimuli_masks, importance_scores):
    # print(img)
    torch.manual_seed(seed)
    ratio = min(height / img.height, width / img.width)
    # img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    if denoise is False:
        strength = 1
    num_denoise_num = math.trunc(num_inference_steps * strength)
    num_start = num_inference_steps - num_denoise_num
    # create the CAC controller.
    # local_blend = LocalBlend(thresh_e=thresh_e, thresh_m=thresh_m, save_inter=False)
    local_blend = LocalBlend(thresh_e=thresh_e, thresh_m=thresh_m, emotion_stimuli_objs=emotion_stimulus_lst,
                             emotion_stimuli_masks=emotion_stimuli_masks,
                             importance_scores=importance_scores, save_inter=True)
    controller = AttentionRefine([source_prompt, target_prompt], [[local, mutual]],
                                 num_inference_steps,
                                 num_start,
                                 cross_replace_steps=cross_replace_steps,
                                 self_replace_steps=self_replace_steps,
                                 local_blend=local_blend
                                 )
    ptp_utils.register_attention_control(pipe, controller)

    results = pipe(prompt=target_prompt,
                   source_prompt=source_prompt,
                   positive_prompt=positive_prompt,
                   negative_prompt=negative_prompt,
                   image=img,
                   num_inference_steps=num_inference_steps,
                   eta=1,
                   strength=strength,
                   guidance_scale=guidance_t,
                   source_guidance_scale=guidance_s,
                   denoise_model=denoise,
                   callback=controller.step_callback
                   )

    # return replace_nsfw_images(results)
    return results.images, False

def generate_src_prompts(image, seed_val):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    if args.do_mood:
        Instruction_COT = \
            f'give 3 concrete emotion visual stimuli of this image, more than 2 words, less than 5 words in each, must include an adjective and a noun; ' \
            f'and one emotional sentence to describe each concrete emotion visual stimuli, more than 20 words in each; ' \
            f'and then describe this entire image in one sentence;' \
            f'and the overall emotion evoked in one sentence.' \
            'your answer should strictly follow this template: ' \
            '''
            Emotion Visual Stimuli and Descriptions:
               1. <Concrete Emotion Visual Stimuli 1 (more than 2 words, less than 5 words)>:
                   Emotional Sentence: "<Emotional Sentence to describe concrete Emotion Visual Stimuli 1 in detail (more than 20 words)>"
               2. <Concrete Emotion Visual Stimuli 2 (more than 2 words, less than 5 words)>:
                   Emotional Sentence: "<Emotional Sentence to describe concrete Emotion Visual Stimuli 2 in detail (more than 20 words)>"
               3. <Concrete Emotion Visual Stimuli 3 (more than 2 words, less than 5 words)>:
                   Emotional Sentence: "<Emotional Sentence to describe concrete Emotion Visual Stimuli 3 in detail (more than 20 words)>"

            Image Description:
               <Image description sentence>.

            Overall Emotion:
               <Overall emotion evoked sentence>
            '''
    elif args.do_face:
        Instruction_COT = \
            '''
            Analyze the facial features following this exact format:\n\n
            1. <Upper Face Features (eyes, eyebrows)>:\n
               Emotional Sentence: "<Detailed description of how eyes and eyebrows convey emotion (more than 20 words)>"\n\n
            2. <Mid Face Features (nose, cheeks)>:\n
               Emotional Sentence: "<Detailed description of how nose and cheeks express emotion (more than 20 words)>"\n\n
            3. <Lower Face Features (mouth, lips)>:\n
               Emotional Sentence: "<Detailed description of how mouth and lips display emotion (more than 20 words)>"\n\n
            Image Description:\n
               <Single sentence comprehensively describing all facial features and their emotional expression>\n\n
            Overall Emotion:\n
               <Single sentence describing the collective emotional impact conveyed by all facial features>
            '''
    conv_template = "llava_llama_3"  # Make sure you use correct chat template for different models

    question = DEFAULT_IMAGE_TOKEN + f"\n{Instruction_COT}\n"  # Jai: add second Image token in the question. By default there is only one.
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question,
        model_loader.llava_tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0)

    # Process image
    image_tensor = process_images([image], model_loader.image_processor, model_loader.llava_model.config)
    image_tensor = [img.to(dtype=torch.float16, device=device2) for img in image_tensor]
    # print('input for generate_src_prompts, image_tensor: ', image_tensor[0].shape)
    # Generate prompts
    output = model_loader.llava_model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.shape],
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
    )
    # print('output for generate_tgt_prompts, output: ', output.shape)
    text_output = model_loader.llava_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # remove all empty lines
    text_output = "\n".join([line for line in text_output.split("\n") if line.strip()])
    text_output = text_output.replace('assistant', '')
    # print('output for generate_tgt_prompts, text_output: ', text_output)
    # print('==============================================================')
    try:
        return text_output.split('Emotion Visual Stimuli and Descriptions:')[1].strip()
    except:
        return text_output.strip().replace('```', '')

def generate_tgt_prompts(image, emotion_question, seed_val):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    if 'make' in emotion_question:
        tmp = 'make ' + emotion_question.split('make')[1].replace('?', '').strip()
    else:
        tmp = 'make this feel more ' + emotion_question.split(' ')[-1].replace('?', '').strip()

    if args.do_mood:
        Instruction_COT = \
            f'''
            Given the image, {emotion_question}? Add/remove/replace objects accordingly, 
            generate a contrasting description to answer the question following this exact format:

        Emotion Visual Stimuli and Descriptions to {tmp}:
            1. Emotion Visual Stimuli 1 to {tmp}: <(concrete objects, more than 2 words, less than 5 words)>:
                Emotional Sentence: '<Emotional description contrasting with original to {tmp} (more than 20 words)>'
            2. Emotion Visual Stimuli 2 to {tmp}: <(concrete objects, more than 2 words, less than 5 words)>:
                Emotional Sentence: '<Emotional description contrasting with original to {tmp} (more than 20 words)>'
            3. Emotion Visual Stimuli 3 to {tmp}: <(concrete objects, more than 2 words, less than 5 words)>:
                Emotional Sentence: '<Emotional description contrasting with original to {tmp} (more than 20 words)>'

        Image Description:
            <Sentence describing how to {tmp}>

        Overall Emotion:
            <Sentence describing how to {tmp}>
            '''
    elif args.do_face:
        Instruction_COT = \
            f''' 
        Given the image, {emotion_question}? 
        generate a contrasting description to answer the question following this exact format:
        1. <Upper Face Features (eyes, eyebrows)> to {tmp}:
            Emotional Sentence: "<Detailed description of how eyes and eyebrows convey emotion (more than 20 words)>"
        2. <Mid Face Features (nose, cheeks)> to {tmp}:
            Emotional Sentence: "<Detailed description of how nose and cheeks express emotion (more than 20 words)>"
        3. <Lower Face Features (mouth, lips)> to {tmp}:
            Emotional Sentence: "<Detailed description of how mouth and lips display emotion (more than 20 words)>"
        Image Description:
            <Sentence of all facial features and their emotional expression describing how to {tmp}>
        Overall Emotion:
            <Sentence describing the collective emotional impact conveyed by all facial features to {tmp}>
        '''

    prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"
    # prompt += f"Look at this image and answer: {emotion_question} add/remove/replace objects accordingly\n\n"
    # prompt += format_instructions + "\n\n"
    # prompt += analysis_instructions
    prompt += Instruction_COT

    conv = copy.deepcopy(conv_templates["llava_llama_3"])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    # print('input for generate_tgt_prompts, prompt_question: ', prompt_question)

    input_ids = tokenizer_image_token(
        prompt_question,
        model_loader.llava_tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0)

    # Process image
    image_tensor = process_images([image], model_loader.image_processor, model_loader.llava_model.config)
    image_tensor = [img.to(dtype=torch.float16, device=device2) for img in image_tensor]
    # print('input for generate_tgt_prompts, image_tensor: ', image_tensor[0].shape)
    # Generate prompts
    output = model_loader.llava_model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.shape],
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
    )
    # print('output for generate_tgt_prompts, output: ', output.shape)
    text_output = model_loader.llava_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # remove all empty lines
    text_output = "\n".join([line for line in text_output.split("\n") if line.strip()])
    text_output = text_output.replace('assistant', '')
    # print('output for generate_tgt_prompts, text_output: ', text_output)
    # print('////////')
    try:
        return text_output.split('Emotion Visual Stimuli and Descriptions:')[1].strip().replace('```', '')
    except:
        return text_output.strip().replace('```', '')

def get_emo_stimulus_importance(image_path, emotion_question):
    instructions = f"""Analyze this image and rate how important changing each object you can detect would be for {emotion_question}, on a scale of 1-10:


        10: Critical impact - Change would completely transform the emotional tone
        8-9: High impact - Major emotional transformation
        6-7: Moderate impact - Noticeable but not dramatic change
        4-5: Low impact - Minor emotional effect
        1-3: Minimal impact - Very little emotional change

        For each element, provide:
        1. Current emotional impact (negative/neutral/positive)
        2. Specific potential for positive transformation
        3. Visual prominence (dominant/moderate/subtle)

        Be precise and consistent in your ratings based on:
        - How much the element currently affects mood (30%)
        - How much changing it would improve joy (40%)
        - How visually prominent it is (30%)

        Format your response as JSON with structure:
        {{
            "<the-object-you-detected>": {{
                "score": number,
                "explanation": "text"
            }}
        }}"""

    conv_template = "llava_llama_3"  # Make sure you use correct chat template for different models

    # question = DEFAULT_IMAGE_TOKEN + f"\n{Instruction_COT}\n"  # Jai: add second Image token in the question. By default there is only one.
    # Construct the prompt
    prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"
    prompt += f"Look at this image and answer: {emotion_question}\n\n"
    prompt += instructions

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question,
        model_loader.llava_tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0)
    # image = load_512("images/81.312_pink-bottle-with-pink-rose-still-life.jpg")
    image = load_512(image_path)
    image_tensor = process_images([image], model_loader.image_processor,
                                  model_loader.llava_model.config)

    # print('image_tensor shape:', image_tensor.shape)
    image_tensor_list = [_image.to(dtype=torch.float16, device=device) for _image in
                         image_tensor]  # Jai: replace image_tensor by image_tensor_list that will be used to append second image

    cont = model_loader.llava_model.generate(
        input_ids,
        images=image_tensor_list,
        image_sizes=[image.shape],
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
    )
    text_outputs = model_loader.llava_tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

    # Clean up the text
    text_output = text_outputs.split('In summary')[0]  # Assuming this is the first output
    cleaned_text = text_output.replace('\\n', '').replace('\n', '').strip()
    cleaned_text = cleaned_text.strip('\'\"')
    cleaned_text = cleaned_text.replace("```json", "").split("```")[0].strip()
    # print(cleaned_text)
    try:
        # Parse the JSON string
        parsed_dict = json.loads(cleaned_text)
        # Get all elements dynamically
        elements = parsed_dict['element']
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        elements = parsed_dict

    # Sort elements by importance score
    sorted_elements = sorted(
        elements.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )

    # Print analysis results
    print("\nEmotional Transformation Analysis:")
    print("=================================")

    saved_dct = defaultdict(int)
    for element, data in sorted_elements:
        print(f"\n{element} (Impact Score: {data['score']}/10)")
        saved_dct[element] = data['score']

    return saved_dct

def get_averaged_importance(image_path, emotion_question, num_runs=3):
    scores = defaultdict(list)

    # Run multiple times and collect scores
    for _ in range(num_runs):
        result = get_emo_stimulus_importance(image_path, emotion_question)
        for element, score in result.items():
            scores[element].append(score)

    # Average the scores
    averaged_scores = {
        element: sum(element_scores) / len(element_scores)
        for element, element_scores in scores.items()
    }

    # Calculate standard deviation to check consistency
    std_devs = {
        element: np.std(element_scores)
        for element, element_scores in scores.items()
    }

    # Print consistency metrics
    print("\nScore Consistency Analysis:")
    for element, avg_score in averaged_scores.items():
        print(f"{element}: {avg_score:.1f} ± {std_devs[element]:.2f}")

    return averaged_scores

def create_progression_video(image_list, fps=4):
    """Convert list of PIL images to a video file"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        video_path = tmp_file.name
        # frames = [np.array(img) for img in image_list]
        frames = []
        for img in image_list:
            if isinstance(img[0], Image.Image):
                frames.append(np.array(img[0]))
            elif isinstance(img, Image.Image):
                frames.append(np.array(img))
            else:
                print("Skip invalid items")
                print(type(img))
                print(img)
                continue
        writer = imageio.get_writer(video_path, fps=fps)

        # Add each frame
        for frame in frames:
            writer.append_data(frame)
        # Hold the last frame for 2 seconds
        for _ in range(fps * 2):
            writer.append_data(frames[-1])

        writer.close()

    return video_path

def remove_repeating_sections(text):
    # Split into lines
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    unique_lines = []

    for i, line in enumerate(cleaned_lines):
        # print('**************', line)
        if line not in unique_lines:
            is_partial = False

            if len(unique_lines) > 0:
                for ul in unique_lines:
                    if line in ul:
                        print(f"Partial duplicate line found at line {i + 1}")
                        is_partial = True
                        break
            if not is_partial:
                unique_lines.append(line)
        else:
            print(f"Duplicate line found at line {i + 1}")

    return '\n'.join(unique_lines)

def clean_text(text, max_words=180):
    # Split into lines and remove empty ones
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Find the "Overall Emotion" section and limit to 2 sentences after it
    overall_emotion_idx = None
    for i, line in enumerate(lines):
        if 'Overall Emotion:' in line:
            overall_emotion_idx = i
            break

    if overall_emotion_idx is not None:
        # Get text after "Overall Emotion:"
        emotion_text = ' '.join(lines[overall_emotion_idx:])
        # Split into sentences (looking for '. ' to avoid splitting decimal numbers)
        sentences = emotion_text.split('. ')
        # Keep only first two sentences after "Overall Emotion:"
        if len(sentences) > 2:
            emotion_text = '. '.join(sentences[:2]) + '.'
            # Replace the original text with truncated version
            lines = lines[:overall_emotion_idx] + [emotion_text]

    # Remove headers we don't want
    lines_ = []
    for line in lines:
        line = line.replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '')
        line = line.replace('Emotional Sentence:', '').replace('Image Description:', '').replace(
            'Overall Emotion:', '')
        line = line.replace('Upper Face Features (eyes, eyebrows):', '').replace('Mid Face Features (nose, cheeks):',
                                                                                 '').replace(
            'Lower Face Features (mouth, lips):', '')
        if 'Emotion Visual Stimuli' in line:
            line = line.split(':')[1].strip()
        line = line.replace('\n', ' ')
        line = line.replace('"', '')
        lines_.append(line)

    # Convert to lowercase
    lines = [line.lower() for line in lines_]

    # Rebuild text with proper formatting
    cleaned_text = ' '.join(lines)
    cleaned_text = cleaned_text.replace("  ", ' ').strip()

    # Check word count and truncate if necessary
    words = cleaned_text.split()
    if len(words) > max_words:
        cleaned_text = ' '.join(words[:max_words]) + '...'

    return cleaned_text

def extract_and_format_all(text):
    # Split text into lines
    lines = text.split('\n')

    stimuli = []
    details = []
    image_desc = ""
    overall_emotion = ""

    current_section = ""

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # For numbered points
        if line.strip() and 'Emotion Visual Stimuli' in line:
            if ': ' in line:
                stimuli.append(line.split(': ')[1].strip())

        # For emotional sentences
        elif line.strip().startswith('Emotional Sentence: '):
            details.append(line.split('Emotional Sentence: ')[1].strip().strip('"'))

        # Track sections
        else:
            image_desc += line.strip() + " "

    # Combine stimuli into one sentence
    combined_stimuli = " ".join(stimuli)

    # Format the output
    result = f"{combined_stimuli}\n"
    for i, detail in enumerate(details, 1):
        result += f"{i}. {detail}\n"

    # Add image description and overall emotion
    # result += f"{image_desc.strip()}\n"

    return result

if args.use_gradio:
    SEED = 0
    EMOTIONS = ['admirable', 'amusing', 'angry', 'annoying', 'approving', 'caring',
                'confusing', 'curious', 'desirable', 'disappointing', 'disapproving', 'disgusting',
                'embarrassing', 'exciting', 'grateful', 'grief', 'joyful', 'loving',
                'nervous', 'optimistic', 'proud', 'realization', 'relief', 'remorse',
                'sad', 'scary', 'surprising']

    emotion_fashion_dict = {
        'admirable': 'elegant, sophisticated, refined',
        'amusing': 'playful, quirky, whimsical',
        'angry': 'bold, edgy, aggressive',
        'annoying': 'clashing, mismatched, chaotic',
        'approving': 'polished, well-coordinated, harmonious',
        'caring': 'comfortable, cozy, nurturing',
        'confusing': 'avant-garde, experimental, unconventional',
        'curious': 'eclectic, innovative, unique',
        'desirable': 'luxurious, alluring, fashionable',
        'disappointing': 'dull, unflattering, outdated',
        'disapproving': 'conservative, restrictive, severe',
        'disgusting': 'tacky, gaudy, excessive',
        'embarrassing': 'ill-fitting, awkward, dated',
        'exciting': 'dynamic, vibrant, statement-making',
        'grateful': 'modest, balanced, appreciative',
        'grief': 'dark, somber, muted',
        'joyful': 'bright, flowing, uplifting',
        'loving': 'romantic, soft, embracing',
        'nervous': 'busy, fussy, over-detailed',
        'optimistic': 'fresh, light, upbeat',
        'proud': 'regal, structured, commanding',
        'realization': 'transformative, revealing, eye-opening',
        'relief': 'relaxed, fluid, easy',
        'remorse': 'heavy, constrained, subdued',
        'sad': 'drooping, loose, unstructured',
        'scary': 'dramatic, intense, intimidating',
        'surprising': 'unexpected, unconventional, innovative'
    }
    emotion_jewlery_dict = {
        'admirable': 'timeless, masterful, exquisite',
        'amusing': 'novelty, funky, playful',
        'angry': 'sharp, angular, aggressive',
        'annoying': 'noisy, cluttered, overwhelming',
        'approving': 'classic, balanced, refined',
        'caring': 'meaningful, personal, heartfelt',
        'confusing': 'abstract, puzzling, complex',
        'curious': 'intricate, detailed, fascinating',
        'desirable': 'precious, coveted, stunning',
        'disappointing': 'plain, uninspired, basic',
        'disapproving': 'harsh, stern, rigid',
        'disgusting': 'garish, over-the-top, tasteless',
        'embarrassing': 'cheap-looking, juvenile, tacky',
        'exciting': 'bold, eye-catching, dynamic',
        'grateful': 'delicate, thoughtful, sincere',
        'grief': 'heavy, dark, substantial',
        'joyful': 'sparkling, lively, energetic',
        'loving': 'romantic, soft, embracing',
        'nervous': 'delicate, fragile, unstable',
        'optimistic': 'light, airy, uplifting',
        'proud': 'statement, grand, impressive',
        'realization': 'clear, crystalline, revealing',
        'relief': 'simple, clean, uncluttered',
        'remorse': 'weighty, serious, substantial',
        'sad': 'dull, tarnished, worn',
        'scary': 'gothic, dark, mysterious',
        'surprising': 'unusual, unexpected, distinctive'
    }
    emotion_vase_dict = {
        'admirable': 'elegant, porcelain, classic',
        'amusing': 'quirky, cartoon-like, whimsical',
        'angry': 'aggressive, sharp, volcanic',
        'annoying': 'noisy, cluttered, overwhelming',
        'approving': 'classic, balanced, refined',
        'caring': 'meaningful, personal, heartfelt',
        'confusing': 'abstract, puzzling, complex',
        'curious': 'intricate, detailed, fascinating',
        'desirable': 'precious, coveted, stunning',
        'disappointing': 'plain, uninspired, basic',
        'disapproving': 'harsh, stern, rigid',
        'disgusting': 'garish, over-the-top, tasteless',
        'embarrassing': 'cheap-looking, juvenile, tacky',
        'exciting': 'bold, eye-catching, dynamic',
        'grateful': 'delicate, thoughtful, sincere',
        'grief': 'heavy, dark, substantial',
        'joyful': 'sparkling, lively, energetic',
        'loving': 'romantic, soft, embracing',
        'nervous': 'delicate, fragile, unstable',
        'optimistic': 'light, airy, uplifting',
        'proud': 'statement, grand, impressive',
        'realization': 'clear, crystalline, revealing',
        'relief': 'simple, clean, uncluttered',
        'remorse': 'weighty, serious, substantial',
        'sad': 'dull, tarnished, worn',
        'scary': 'gothic, dark, mysterious',
        'surprising': 'unusual, unexpected, distinctive'
    }

    QUESTION_TEMPLATES = [
        "how could we make this feel more {emotion}?",
        "how to make this image more {emotion}?",
        "could you make this picture more {emotion}?",
        "how would you transform this to be more {emotion}?"
    ]

    # First, let's modify your EMOTION_COLORS dictionary to use rgba values
    EMOTION_COLORS = {
        'admiration': '#9c4dcc',  # purple
        'amusing': '#ff9f43',  # orange
        'angry': '#ee5253',  # red
        'annoying': '#786fa6',  # brown-gray
        'approval': '#7bed9f',  # green
        'caring': '#ff6b81',  # pink
        'confusing': '#a4b0be',  # gray
        'curious': '#48dbfb',  # light blue
        'desire': '#ff6b81',  # pink
        'disappointing': '#a4b0be',  # gray
        'disapproval': '#800000',  # dark red
        'disgust': '#341f97',  # dark purple
        'embarrassing': '#ff7f50',  # coral
        'exciting': '#ffd32a',  # yellow
        'grateful': '#7bed9f',  # green
        'grief': '#1e3799',  # navy blue
        'joy': '#ffd32a',  # yellow
        'love': '#ff6b81',  # pink
        'nervous': '#a4b0be',  # gray
        'optimistic': '#ffd32a',  # yellow
        'proud': '#9c4dcc',  # purple
        'realization': '#48dbfb',  # light blue
        'relief': '#7bed9f',  # green
        'remorse': '#4b6584',  # dark gray
        'sad': '#3742fa',  # blue
        'scary': '#2f3542',  # dark gray
        'surprise': '#70a1ff',  # light blue
        # 'neutral': '#a4b0be'  # gray
    }

    css = """
      /* Target Gradio's button classes specifically */
      button.lg.secondary.emotion-button {
          border-radius: 9999px !important;
          padding: 8px 16px !important;
          margin: 4px !important;
          font-size: 14px !important;
          transition: all 0.2s ease-in-out !important;
          color: white !important;
          font-weight: 500 !important;
      }

      .circle-button {
      border-radius: 50% !important;
      width: 32px !important;
      height: 32px !important;
      padding: 0 !important;
      min-width: 32px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      }

      .custom-idea-header {
      margin: 0 !important;
      display: inline-block !important;
      }
      .generate-button {
      position: relative !important;
      overflow: visible !important;
      }

      .progress-bar {
     position: absolute !important;
     top: -4px !important;
     left: 0 !important;
     width: 100% !important;
     height: 4px !important;
     overflow: hidden !important;
     background: rgba(255, 255, 255, 0.1) !important;
      }

      .progress-line {
     height: 100% !important;
     background: linear-gradient(90deg, 
         #ff6b6b 0%, 
         #ffd93d 25%,
         #6ce5b1 50%,
         #4cc9f0 75%,
         #ff6b6b 100%) !important;
     background-size: 400% 100% !important;
     animation: 
         progress-move 60s linear forwards,
         shimmer 3s linear infinite,
         pulse 2s ease-in-out infinite !important;
     box-shadow: 
         0 0 15px rgba(76, 201, 240, 0.6),
         0 0 30px rgba(76, 201, 240, 0.3) !important;
      }

      @keyframes progress-move {
         0% { width: 0%; }
         100% { width: 100%; }
      }

      @keyframes shimmer {
         0% { background-position: 400% 0; }
         100% { background-position: -400% 0; }
      }

      @keyframes pulse {
         0%, 100% { transform: scaleY(1); }
         50% { transform: scaleY(1.5); }
      }

      .generate-button::before {
          content: '';
          position: absolute !important;
          top: -2px !important;  /* Positioned just above the button */
          left: 0 !important;
          height: 2px !important;  /* Very thin line */
          width: 0% !important;
          background: linear-gradient(to right, #00ff87, #60efff) !important;  /* Gradient colors */
          transition: width 0.1s ease !important;
          box-shadow: 0 0 8px rgba(96, 239, 255, 0.5) !important;  /* Subtle glow */
          border-radius: 2px !important;
          z-index: 1000 !important;
      }

      .generate-button.processing::before {
          animation: progress-animation 60s linear forwards !important;
      }


      .section-row {
      position: relative;
      display: flex;
      align-items: center;
      gap: 12px;
      }

      .remove-button {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          right: -40px;
          width: 32px !important;
          height: 32px !important;
          border-radius: 50% !important;
          padding: 0 !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
          font-size: 20px !important;
      }

      .add-section {
          margin-top: 20px;
      }

      /* Override specific emotion colors */
      """

    # Add color classes with higher specificity
    for emotion, color in EMOTION_COLORS.items():
        css += f"""
      button.lg.secondary.emotion-{emotion} {{
          background-color: {color} !important;
          border: 3px solid {color} !important;
      }}

      button.lg.secondary.emotion-{emotion}:hover {{
          background-color: {color} !important;
          border: 7px solid {color} !important;
          filter: brightness(1.1) !important;
          transform: translateY(-2px) !important;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
      }}
      """


    def update_selected_emotions_text(emotion, selected_emotions_text, state):
        """Update the selected emotions text while maintaining all selections"""
        # Initialize or get current selections
        current_selections = set(selected_emotions_text.split(", ")) if selected_emotions_text else set()

        # Toggle the current emotion
        if emotion in current_selections:
            current_selections.remove(emotion)
        else:
            current_selections.add(emotion)

        # Return updated text, excluding empty string if present
        return ", ".join(sorted(e for e in current_selections if e))


    def generate_final_question(selected_emotions_text):
        """Generate question only when confirm button is clicked"""
        if not selected_emotions_text:
            return "Please select some emotions first"

        emotions = [(e.strip(), emotion_fashion_dict[e.strip()]) for e in selected_emotions_text.split(",")]
        emotions_text = " and ".join([f"{e[0]} ({e[1]})" for e in emotions])
        template = random.choice(QUESTION_TEMPLATES)
        return template.format(emotion=emotions_text)


    with gr.Blocks(title="MLLM-Enhanced Emotion-Driven Image Editing", css=css) as interface:
        class SharedState:
            def __init__(self):
                self.raw_text_output = None
                self.source_prompt = None
                self.target_prompt = None
                self.source_prompt_raw = None
                self.target_prompt_raw = None
                self.valid_objs = None
                self.valid_masks = None
                self.importance_tuple = None
                self.last_image = None
                self.last_emotion_q = None
                self.processed_image = None
                self.obj_to_mask = {}
                self.selected_emotions = []


        shared_state = SharedState()

        gr.Markdown("# Moodifier: MLLM-Enhanced Emotion-Driven Image Editing")
        emotion_buttons = []

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="pil"
                )

                with gr.Row():
                    show_emotions_btn = gr.Button("Choose Emotion", variant="primary")
                    # The question textbox
                    emotion_question = gr.Textbox(
                        label="Emotion Question",
                        placeholder="Question will appear after confirming emotions",
                        value="",
                        lines=1,
                        interactive=True
                    )
                with gr.Row():
                    # Custom question input
                    custom_question = gr.Textbox(
                        label="Or type your own question",
                        placeholder="e.g., How could we make this image more vibrant?",
                        lines=1
                    )
                # Emotion selection container
                with gr.Column(visible=False) as emotion_container:
                    gr.Markdown("## Select Multiple Emotions")

                    with gr.Row():
                        # Display selected emotions
                        selected_emotions_display = gr.Textbox(
                            label="Currently Selected Emotions",
                            placeholder="Selected emotions will appear here",
                            value="",
                            lines=1,
                            interactive=False
                        )
                        with gr.Column():
                            confirm_emotions_btn = gr.Button("✓ Confirm Emotions", variant="primary")
                            clear_emotions_btn = gr.Button("Clear Selections", variant="secondary")

                    # Create emotion grid
                    for i in range(0, len(EMOTIONS), 4):
                        with gr.Row():
                            for emotion in EMOTIONS[i:i + 4]:
                                btn = gr.Button(
                                    emotion,
                                    elem_classes=["lg", "secondary", f"emotion-{emotion}"]
                                )
                                # Pass the current selections text to maintain state
                                btn.click(
                                    fn=update_selected_emotions_text,
                                    inputs=[
                                        btn,
                                        selected_emotions_display,  # Pass current selections
                                        gr.State(shared_state)
                                    ],
                                    outputs=[selected_emotions_display]
                                )


                def clear_selections():
                    """Clear all selected emotions"""
                    return "", ""  # Clear both the selections display and the question


                def generate_final_question(selected_emotions_text):
                    """Generate question when confirm button is clicked"""
                    if not selected_emotions_text:
                        return "Please select some emotions first"

                    emotions = [e.strip() for e in selected_emotions_text.split(",") if e.strip()]
                    shared_state.selected_emotions = emotions
                    emotions_text = " and ".join(emotions)
                    template = random.choice(QUESTION_TEMPLATES)
                    return template.format(emotion=emotions_text)


                def toggle_emotion_grid(state):
                    """Toggle visibility of emotion grid"""
                    if not hasattr(state, 'click_count'):
                        state.click_count = 0

                    state.click_count += 1

                    # On first click, show grid
                    if state.click_count == 1:
                        return gr.update(visible=True)
                    # On second click, hide grid and reset count
                    else:
                        state.click_count = 0
                        return gr.update(visible=False)

                # Connect the show/hide button
                show_emotions_btn.click(
                    fn=toggle_emotion_grid,
                    inputs=[gr.State(shared_state)],
                    outputs=[emotion_container]
                )
                # Connect confirm button
                confirm_emotions_btn.click(
                    fn=generate_final_question,
                    inputs=[selected_emotions_display],
                    outputs=[emotion_question]
                )
                # Add clear button handler
                clear_emotions_btn.click(
                    fn=clear_selections,
                    inputs=[],
                    outputs=[selected_emotions_display, emotion_question]
                )

                # Connect custom question changes to clear emotion selections
                custom_question.change(
                    fn=lambda: ("", ""),  # Clear emotion selections when user types custom question
                    inputs=[],
                    outputs=[selected_emotions_display, emotion_question]
                )

                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,
                    label="Guidance Scale"
                )
                # num_runs = gr.Slider(
                #     minimum=1,
                #     maximum=5,
                #     value=1,
                #     step=1,
                #     label="Number of Runs for Importance Analysis"
                # )
                # Add seed slider after guidance_scale
                seed = gr.Slider(
                    minimum=0,
                    maximum=1000000,
                    value=1,
                    step=1,
                    label="Random Seed"
                )

                # Raw output display (editable)
                # raw_output_display = gr.Textbox(
                #     label="Raw LLaVA Output (Edit if needed)",
                #     lines=10,
                #     interactive=True
                # )
                # Regular prompt displays (not editable)
                # source_prompt_display = gr.Textbox(
                #     label="Source Prompt (Edit if needed)",
                #     lines=5,
                #     interactive=True  # Changed to False
                # )
                # analyze_button = gr.Button("Generate Detailed Prompts for the Required Emotion")

            with gr.Column():
                target_prompt_display = gr.Textbox(
                    label="Target Prompt",
                    lines=5,
                    interactive=True  # Changed to False
                )
                # Display objects and scores side by side
                with gr.Row():
                    analyze_button = gr.Button("Generate Detailed Prompts for the Required Emotion")
                    save_raw_button = gr.Button("Save My Edits")

                with gr.Row():
                    with gr.Column():
                        objects_display = gr.Textbox(
                            label="Detected Objects",
                            lines=5,
                            interactive=True
                        )
                    with gr.Column():
                        scores_display = gr.Textbox(
                            label="Importance Scores",
                            lines=5,
                            interactive=True
                        )
                process_edits_button = gr.Button("Locate Emotional Stimulus")

                # Add new object/score inputs
                with gr.Row():
                    with gr.Column():
                        new_object = gr.Textbox(
                            label="Add Object",
                            placeholder="e.g.: rose",
                            lines=1
                        )
                    with gr.Column():
                        new_score = gr.Textbox(
                            label="Add Score",
                            placeholder="e.g.: 7.00",
                            lines=1
                        )
                    with gr.Column():
                        add_button = gr.Button("Add")
                        update_button = gr.Button("Update Stimulus Attention")

                edit_button = gr.Button("Edit Image")
                clear_button = gr.Button("Clear Analysis Cache")

        with gr.Row():
            # video_output = gr.Video(
            #     label="Editing Progression",
            #     format="mp4"
            # )
            gallery_output_attnmasks = gr.Gallery(
                label="Attention Masks",
                columns=2,
                rows=2,
                height="auto"
            )
            gallery_output_edits = gr.Gallery(
                label="Final Results",
                columns=2,
                rows=2,
                height="auto"
            )


        # def update_raw(src_text, tgt_text):
        #     shared_state.source_prompt_raw = src_text
        #     shared_state.target_prompt_raw = tgt_text
        #     shared_state.source_prompt = clean_text(src_text)
        #     shared_state.target_prompt = clean_text(tgt_text)
        def update_raw(tgt_text):
            shared_state.target_prompt_raw = tgt_text
            text_tgt_output = remove_repeating_sections(tgt_text)
            text_tgt_output = extract_and_format_all(text_tgt_output)
            shared_state.target_prompt = clean_text(text_tgt_output)

            gr.Info("Your Edits have been safed successfully!")


        def analyze_image(image, emotion_q, seed_val):
            try:
                gr.Info("Starting image analysis...")

                if image is None:
                    raise gr.Error("Please upload an image first!")

                processed_image = load_512(image)
                shared_state.processed_image = processed_image
                shared_state.last_image = image
                shared_state.last_emotion_q = emotion_q

                print(emotion_q)
                # Get raw LLaVA output
                # text_src_output = generate_src_prompts(processed_image, seed_val)  # Get only text_output

                if 'exciting' in shared_state.selected_emotions:
                    text_tgt_output = \
'''
Emotion Visual Stimuli and Descriptions to make this feel more exciting:

Emotion Visual Stimuli 1 to make this feel more exciting: metallic gold asymmetrical shirt:
    Emotional Sentence: 'A dynamic metallic gold shirt with dramatic asymmetrical lapels that catch and reflect light with every movement, featuring vibrant electric blue accent piping and statement shoulders that create a powerful silhouette impossible to ignore.'
Emotion Visual Stimuli 2 to make this feel more exciting: iridescent cutout leather pants:
    Emotional Sentence: 'High-impact leather pants featuring strategic cutouts lined with iridescent material that shifts color depending on viewing angle, accentuated by vibrant contrast stitching and hardware that creates rhythmic visual movement throughout the lower body.'
Emotion Visual Stimuli 3 to make this feel more exciting: light-up liquid platform shoes:
    Emotional Sentence: 'Bold platform shoes with clear heels filled with swirling neon liquid, flashing LED lights embedded along the sides that pulse with music, metallic gold and electric blue color-blocking that catches every eye, and an extra 5 inches of height that makes the wearer tower dramatically above the crowd.'

Image Description:
    Reinvent this understated outfit as an electrifying ensemble through high-contrast elements, kinetic details, and unexpected texture combinations that create visual rhythm and constantly draw the eye to different aspects of the complete look.
Overall Emotion:
    This transformed look creates genuine excitement through its dynamic, vibrant, and statement-making elements that blend bold structural choices with unexpected details, transforming a simple outfit into a multisensory experience that pulses with energy and possibility.
'''
                elif 'angry' in shared_state.selected_emotions:
                    text_tgt_output = \
'''
Emotion Visual Stimuli and Descriptions to make this feel more angry:

Emotion Visual Stimuli 1 to make this feel more angry: spiked black leather crop shirt:
    Emotional Sentence: 'A bold, distressed black leather crop shirt with threatening metal spikes protruding from the shoulders and deliberate slash marks revealing blood-red fabric beneath, creating an armor-like garment that actively challenges anyone who dares approach.'
Emotion Visual Stimuli 2 to make this feel more angry: battle-ready chain pants:
    Emotional Sentence: 'Edgy battle-ready pants featuring aggressive straps and chains that clank ominously with movement, with strategic tears reinforced by metal grommets and a deep crimson accent stripe down each leg that commands space like a warning sign.'
Emotion Visual Stimuli 3 to make this feel more angry: metal-spiked combat shoes:
    Emotional Sentence: 'Extreme combat shoes with 3-inch metal spikes covering every surface, flame decals painted along the sides, thick chains wrapped around the ankles, steel-plated toes perfect for kicking, and heavy treads that stomp so loudly they sound like thunder with each aggressive step.'

Image Description:
    Transform this conventional outfit into a confrontational ensemble through aggressive textures, sharp metallic details, and a deliberately destructive aesthetic that communicates intense emotional power and readiness for conflict.
Overall Emotion:
    This transformed look radiates controlled fury through its bold, edgy, and aggressive elements that transform clothing into a visual declaration of strength that demands respect and creates deliberate distance from conformist fashion sensibilities.
'''
                elif 'amusing' in shared_state.selected_emotions:
                    text_tgt_output = \
'''
Emotion Visual Stimuli and Descriptions to make this feel more amusing:

Emotion Visual Stimuli 1 to make this feel more amusing: banana peel trompe loeil shirt:
    Emotional Sentence: 'A playful trompe loeil design that appears to be a banana peel draped over the shoulders, featuring whimsical oversized buttons shaped like cartoon monkey faces and sleeves that unexpectedly inflate like balloon animals when the wearer gestures enthusiastically.'
Emotion Visual Stimuli 2 to make this feel more amusing: mismatched quirky yellow pants:
    Emotional Sentence: 'Quirky pants with mismatched legs—one shorter than the other—featuring comically oversized pockets placed in deliberately impractical locations and a waistband that plays a cheerful tune whenever someone compliments the outfit.'
Emotion Visual Stimuli 3 to make this feel more amusing: rubber duck shoes:
    Emotional Sentence: 'Giant rubber duck shoes that squeak loudly with every step, featuring bright yellow bodies, oversized orange beaks as toe caps, googly eyes that wobble as you walk, and comically large webbed feet that slap against the floor creating a waddling gait impossible to ignore or take seriously.'

Image Description:
    Reimagine this sophisticated outfit as a playful ensemble that embraces childlike joy through unexpected interactive elements, deliberately mismatched proportions, and quirky details that prioritize fun over function or conventional aesthetics.
Overall Emotion:
    This transformed look creates genuine amusement through its playful, quirky, and whimsical approach to fashion that transforms everyday clothing into an interactive experience designed to spark joy and conversation.
'''
                else:
                    text_tgt_output = generate_tgt_prompts(processed_image, emotion_q, seed_val)  # Get only text_output

                # shared_state.source_prompt_raw = text_src_output
                shared_state.target_prompt_raw = text_tgt_output

                # text_src_output = remove_repeating_sections(text_src_output)
                text_tgt_output = remove_repeating_sections(text_tgt_output)

                # if len(text_src_output.split()) > len(text_tgt_output.split()):
                # img_desc_start = text_src_output.find("Image Description:")
                # if img_desc_start != -1:
                #     Keep only from Image Description onwards
                    # text_src_output = text_src_output[img_desc_start:]

                text_tgt_output = extract_and_format_all(text_tgt_output)

                # shared_state.source_prompt = clean_text(text_src_output)
                shared_state.target_prompt = clean_text(text_tgt_output)
                # shared_state.raw_text_output = text_output
                # shared_state.user_edited_raw = text_output
                # print('shared_state.raw_text_output: ', shared_state.raw_text_output)

                # Add 5 second delay
                time.sleep(5)

                gr.Info("Analysis complete! You can edit the raw output if needed.")
                # yield text_tgt_output
                yield shared_state.target_prompt_raw

            except Exception as e:
                raise gr.Error(f"Analysis failed: {str(e)}")


        def process_edited_text(source_prompt, target_prompt):
            num_runs = 1
            try:
                gr.Info("Processing edited text...")
                logger.debug("Starting to process edited text")

                if args.do_face:
                    shared_state.importance_tuple = {
                        'eyes': 4.0,
                        'eyebrows': 3.0,
                        'mouth': 4.0,
                        'nose': 2.0,
                        'face': 5.0
                    }
                elif args.do_mood:
                    # Get importance scores for each object
                    # shared_state.importance_tuple = get_averaged_importance(
                    #     shared_state.processed_image,
                    #     # valid_objs,
                    #     shared_state.last_emotion_q,
                    #     num_runs=num_runs  # or whatever num_runs value you want
                    # )
                    shared_state.importance_tuple = {
                        # 'sunglasses': 8.0,
                        'shirt': 8.0,
                        # 'belt': 8.0,
                        # 'pants': 7.0,
                        'pants': 7.0,
                        'shoes': 5.0
                        # 'shirt': 8.0
                    }

                valid_objs = []
                valid_masks = []

                # shared_state.importance_tuple['background'] = 10.00

                for obj in list(shared_state.importance_tuple.keys()):
                    # res_mask, _ = segment_emo_stimuli(shared_state.processed_image, obj)
                    # res_mask = prompt2mask(shared_state.processed_image, obj)
                    res_mask = prompt2mask(shared_state.processed_image, obj, box_threshold=0.335)

                    if res_mask.max() > 0:
                        valid_objs.append(obj)
                        valid_masks.append(res_mask)

                        if obj not in shared_state.obj_to_mask:
                            shared_state.obj_to_mask[obj] = res_mask

                shared_state.valid_objs = valid_objs
                shared_state.valid_masks = valid_masks

                # obj_importance_text = "\n".join(
                #     [f"{obj}: {shared_state.importance_tuple[obj]:.2f}" for obj in valid_objs])

                # obj_list = []
                # score_list = []
                # for obj in valid_objs:
                #     obj_list.append(obj)
                #     score_list.append(f"{shared_state.importance_tuple[obj]:.2f}")
                #
                # gr.Info("Text processed successfully!")
                # logger.debug("Text processing complete")
                # yield "\n".join([i for i in obj_list if i != 'background']), "\n".join(
                #     [i for i in score_list if i != '10.00'])

                # Prepare all results
                results = []

                # First entry: all objects and scores together
                all_objs = []
                all_scores = []
                for obj in valid_objs:
                    if obj != 'background':
                        all_objs.append(obj)
                        all_scores.append(f"{shared_state.importance_tuple[obj]:.2f}")

                results.append(("\n".join(all_objs), "\n".join(all_scores)))

                # Then add individual entries for each object
                for obj in valid_objs:
                    if obj != 'background':
                        results.append((obj, f"{shared_state.importance_tuple[obj]:.2f}"))

                gr.Info("Text processed successfully!")
                logger.debug("Text processing complete")

                # Yield all results as a list
                yield results[0]

            except Exception as e:
                logger.error(f"Error in process_edited_text: {str(e)}", exc_info=True)
                raise gr.Error(f"Failed to process edited text: {str(e)}")


        def edit_image(image, g_scale, seed_val, objects, scores):
            # try:
            #
            # except Exception as e:
            #     logger.error(f"Error in edit_image: {str(e)}", exc_info=True)
            #     raise gr.Error(f"Editing failed: {str(e)}")
            gr.Info("Starting image editing...")
            logger.debug("Starting edit_image function")
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            random.seed(seed_val)

            if isinstance(image, str):
                image_for_edit = image
            else:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    image_for_edit = tmp_file.name

            if not shared_state.valid_objs:
                raise gr.Error("No valid objects detected to edit!")

            # importance_scores = [shared_state.importance_tuple[obj]
            #                      for obj in shared_state.valid_objs]
            # valid_objs = [obj.strip() for obj in objects.split('\n') if obj.strip() and len(obj.strip()) != 0]
            def split_target_prompt(target_prompt):

                content = target_prompt

                # First, find where the objects end and descriptions begin
                parts = content.split("'")
                if len(parts) < 2:
                    return []

                # The first part contains "<obj 1>: <obj 2>: <obj 3>: "
                objects_part = parts[0]

                # Split by colon to get the objects
                objects = [obj.strip() for obj in objects_part.split(':')[:-1]]  # Skip the last empty part

                # Extract descriptions (they're in odd-indexed parts between quotes)
                descriptions = []
                for i in range(1, len(parts), 2):
                    if i < len(parts):
                        descriptions.append(parts[i])

                # Pair each object with its description
                result = []
                for i in range(min(len(objects), len(descriptions))):
                    result.append((objects[i], descriptions[i]))

                return result

            obj_desc_pairs = split_target_prompt(shared_state.target_prompt)

            loaded_image = load_512(image_for_edit, resize=False)
            h, w, c = loaded_image.shape
            print('h, w, c: ', h, w, c)
            video_path_lst = []
            attn_mask_lst = defaultdict()
            final_edited_images = defaultdict()

            tmp_lst = [list(shared_state.obj_to_mask.keys())]
            for k, v in shared_state.obj_to_mask.items():
                tmp_lst.append([k])
            for obj_idx, valid_objs in enumerate(tmp_lst):

                valid_masks = [shared_state.obj_to_mask[obj] for obj in valid_objs]
                importance_scores = [float(score.strip()) for score in scores.split('\n')]

                print('valid_objs: ', len(valid_objs), valid_objs)

                obj_prompt = ''
                desc_prompt = ''
                for obj, desc in obj_desc_pairs:
                    print('obj: ', obj)
                    print('desc: ', desc)
                    if valid_objs[0] in obj:  # Check if valid_obj is part of the object name
                        obj_prompt += obj + '. '
                        desc_prompt += desc + '. '
                target_prompt = obj_prompt + desc_prompt
                print('target_prompt: ', target_prompt)
                edited_images, nsfw_detected = inference(img=Image.fromarray(loaded_image),
                                                         source_prompt='',
                                                         target_prompt=target_prompt,
                                                         local="",
                                                         mutual="",
                                                         positive_prompt="same age, masterpiece, best quality, high quality",
                                                         negative_prompt="different age, text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                                         guidance_s=0.6,
                                                         guidance_t=2.0 if (h >= 800 or w >= 800) else 2.5,
                                                         num_inference_steps=15,
                                                         width=512,
                                                         height=512,
                                                         seed=seed_val,
                                                         strength=1.0,
                                                         cross_replace_steps=0.5 if (h >= 800 or w >= 800) else 0.7,
                                                         self_replace_steps=0.5 if (h >= 800 or w >= 800) else 0.7,
                                                         thresh_e=0.5,
                                                         thresh_m=0.5,
                                                         denoise=True,
                                                         emotion_stimulus_lst=valid_objs,
                                                         emotion_stimuli_masks=valid_masks,
                                                         importance_scores=importance_scores)

                video_path = create_progression_video(edited_images)
                video_path_lst.append(video_path)

                try:
                    attn_mask = Image.open("output1/mask_e/14.jpg")
                    # Resize the attention mask to match the dimensions of the final edited image
                    if attn_mask is not None and edited_images[-1][0] is not None:
                        target_size = edited_images[-1][0].size
                        attn_mask = attn_mask.resize(target_size, Image.LANCZOS)
                except Exception as e:
                    logger.error(f"Error loading attention mask: {str(e)}")
                    attn_mask = None
                attn_mask_lst[' '.join(valid_objs)] = attn_mask
                final_edited_images[' '.join(valid_objs)] = edited_images[-1][0]
                gr.Info("Edit {} complete!".format(obj_idx + 1))
                gr.Info('Attention Mask {} stored!'.format(obj_idx + 1))

            gr.Info("Editing complete!")
            logger.debug("Editing completed successfully")
            # return video_path, \
            #                 [
            #                     (edited_images[-1][0], "Final Edit"),  # Add appropriate captions
            #                     # (edit1, "Edit 1"),
            #                     # (edit2, "Edit 2"),
            #                     (attn_mask, "Attention Mask")
            #                 ]
            return [(v, f'Attention Mask {k}') for k, v in attn_mask_lst.items()], [(v, f'Edit {k}') for k, v in
                                                                                    final_edited_images.items()]


        def clear_state():
            logger.debug("Clearing shared state")
            shared_state.__init__()
            gr.Info("Analysis cache and edits cleared!")
            return None, None, None, None  # For all displays


        def add_object_score(objects, scores, new_obj, new_score):
            new_obj = new_obj.strip()
            if not new_obj.strip() or not new_score.strip():
                return objects, scores

            # Calculate mask for new object
            if new_obj not in shared_state.obj_to_mask:
                # res_mask, _ = segment_emo_stimuli(shared_state.processed_image, new_obj)
                res_mask = prompt2mask(shared_state.processed_image, new_obj)
                shared_state.obj_to_mask[new_obj] = res_mask

            return objects + f"\n{new_obj}", scores + f"\n{new_score}"


        def update_stimulus(objects, scores):
            # Keep track of remaining objects to clean up mask dict
            remaining_objs = set(line.strip() for line in objects.split("\n") if line.strip())

            # Remove unused masks from dict
            shared_state.obj_to_mask = {
                obj: mask for obj, mask in shared_state.obj_to_mask.items()
                if obj in remaining_objs
            }
            for obj in remaining_objs:
                if obj not in shared_state.obj_to_mask.keys():
                    # res_mask, _ = segment_emo_stimuli(shared_state.processed_image, obj)
                    res_mask = prompt2mask(shared_state.processed_image, obj)
                    shared_state.obj_to_mask[obj] = res_mask

            return "\n".join(remaining_objs), scores


        add_button.click(
            fn=add_object_score,
            inputs=[objects_display, scores_display, new_object, new_score],
            outputs=[objects_display, scores_display]
        )

        update_button.click(
            fn=update_stimulus,
            inputs=[objects_display, scores_display],
            outputs=[objects_display, scores_display]
        )

        save_raw_button.click(
            fn=lambda x: update_raw(x),  # Add update_raw method to SharedState class
            inputs=[target_prompt_display],
            outputs=[]
        )

        # Connect components
        analyze_button.click(
            fn=analyze_image,
            inputs=[input_image, emotion_question, seed],
            outputs=[target_prompt_display]
        )

        process_edits_button.click(
            fn=process_edited_text,
            inputs=[target_prompt_display],
            outputs=[objects_display, scores_display]
        )

        edit_button.click(
            fn=edit_image,
            inputs=[input_image, guidance_scale, seed, objects_display, scores_display],
            # outputs=[video_output, gallery_output]
            outputs=[gallery_output_attnmasks, gallery_output_edits]
        )

        clear_button.click(
            fn=clear_state,
            outputs=[target_prompt_display, objects_display, scores_display]
        )

    interface.launch(share=True, debug=False)

elif args.do_face:
    # image_path = 'images/test_images/face/5 (2).jpg'
    # folder_name = 'woman'
    # os.makedirs(f'output1/{folder_name}', exist_ok=True)

    SEED = 0
    # Define input and output paths
    for filename_idx, filename in enumerate(os.listdir('images/test_images/face')):
        if filename.endswith('.jpg') or filename.endswith('.png') or \
            filename.endswith('.jpeg') or filename.endswith('.webp') or \
                filename.endswith('.avif'):
            folder_name = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.webp', '').replace('.avif', '')
            try:
                image_path = os.path.join('images/test_images/face', filename)
                os.makedirs(f'output2_i4/face/{folder_name}', exist_ok=True)

                # Load image once to avoid multiple loads
                loaded_image = load_512(image_path, resize=False)
                h, w, c = loaded_image.shape

                emotion_list = [
                    'admired', 'amused', 'angry', 'annoyed', 'approved', 'caring',
                    'confused', 'curious', 'desire', 'disappointed', 'disapproved',
                    'disgusted', 'embarrassed', 'excited', 'grateful', 'grief', 'joyful',
                    'love', 'nervous', 'optimistic', 'proud', 'realization', 'relief',
                    'remorse', 'sad', 'scary', 'surprised'
                ]
                # emotion_list = ['excited']


                importance_tuple = {
                    'eyes': 8.0,
                    'forehead': 4.0,
                    'mouth': 8.0,
                    'lips': 8.0,
                    'nose': 4.0,
                    'face': 3.0
                }

                valid_objs = []
                valid_masks = []
                # importance_tuple['background'] = 10.00

                for obj in list(importance_tuple.keys()):
                    # res_mask, _ = segment_emo_stimuli(shared_state.processed_image, obj)
                    res_mask = prompt2mask(loaded_image, obj)
                    if res_mask.max() > 0:
                        valid_objs.append(obj)
                        valid_masks.append(res_mask)

                obj_list = []
                score_list = []
                for obj in valid_objs:
                    obj_list.append(obj)
                    score_list.append(importance_tuple[obj])

                for emotion in emotion_list:
                    if os.path.exists(f'output2_i4/face/{folder_name}/{emotion}{SEED}.png'):
                        continue
                    print(f"Processing emotion: {emotion}")
                    # Construct emotion question based on emotion
                    emotion_question = f"how could we make this person look more {emotion}?"

                    # Get automatic prompts using LLaVA
                    # src_prompt = generate_src_prompts(loaded_image, 0)
                    tgt_prompt = generate_tgt_prompts(loaded_image,
                                                      emotion_question,
                                                      SEED)

                    # print("\nGenerated Source Prompt:", src_prompt)
                    # print("Generated Target Prompt:", tgt_prompt)
                    #
                    # src_prompt = remove_repeating_sections(src_prompt)
                    tgt_prompt = remove_repeating_sections(tgt_prompt)

                    # img_desc_start = src_prompt.find("Image Description:")
                    # if img_desc_start != -1:
                    #     # Keep only from Image Description onwards
                    #     src_prompt = src_prompt[img_desc_start:]
                    #
                    print(f"Generated target prompt 1: {tgt_prompt}")

                    # tgt_prompt = extract_and_format_all(tgt_prompt)
                    img_desc_start = tgt_prompt.find("Image Description:")
                    if img_desc_start != -1:
                        # Keep only from Image Description onwards
                        tgt_prompt = tgt_prompt[img_desc_start:]

                    # src_prompt = clean_text(src_prompt)
                    tgt_prompt = clean_text(tgt_prompt)

                    tgt_prompt = tgt_prompt.replace("slightly parted", "wide open showing teeth")
                    # print(f"Generated source prompt: {src_prompt}")
                    print(f"Generated target prompt 2: {tgt_prompt}")

                    # Perform the editing
                    edited_images, nsfw_detected = inference(img=Image.fromarray(loaded_image),
                                                             source_prompt='',
                                                             target_prompt=tgt_prompt,
                                                             local="",
                                                             mutual="",
                                                             positive_prompt="same age, masterpiece, best quality, high quality",
                                                             negative_prompt="different age, text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                                             guidance_s=0.6,
                                                             guidance_t=3.0 if (h > 800 or w > 800) else 2.0,
                                                             num_inference_steps=15,
                                                             width=512,
                                                             height=512,
                                                             seed=SEED,
                                                             strength=1.0,
                                                             cross_replace_steps=0.7 if (h > 800 or w > 800) else 0.8,
                                                             self_replace_steps=0.7 if (h > 800 or w > 800) else 0.8,
                                                             thresh_e=0.3,
                                                             thresh_m=0.3,
                                                             denoise=True,
                                                             emotion_stimulus_lst=valid_objs,
                                                             emotion_stimuli_masks=valid_masks,
                                                             importance_scores=score_list)
                    print('edited_images[-1]: ', len(edited_images[-1]), edited_images[-1][0].size)
                    edited_images[-1][0].save(f'output2_i4/face/{folder_name}/{emotion}{SEED}.png')
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

elif args.do_mood:
    SEED = 1
    # image_path = 'images/test_images/object/img_1546.webp'
    # folder_name = 'dress1'
    # os.makedirs(f'output1/{folder_name}', exist_ok=True)
    for cate in os.listdir('images/test_images/object'):
        if cate != 'clothes':
            continue
        for filename in os.listdir(f'images/test_images/object/{cate}'):
            if not filename.startswith('ac6f42ffea9c7d34b4d3ce026ecfe5df'):
                continue
            if filename.endswith('.jpg') or filename.endswith('.png') or \
                filename.endswith('.jpeg') or filename.endswith('.webp'):
                folder_name = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.webp', '').replace('.avif', '')
                try:
                    image_path = os.path.join(f'images/test_images/object/{cate}', filename)
                    os.makedirs(f'output2_i4_{args.use_clip}/{cate}/{folder_name}', exist_ok=True)
                    # Load image once to avoid multiple loads
                    loaded_image = load_512(image_path, resize=False)
                    h, w, c = loaded_image.shape
                    if cate == 'earring' or cate == 'necklace' or cate == 'clothes' or cate == 'ring' or \
                            cate == 'vase' or cate == 'bracelet' or cate == 'bag':
                        importance_tuple = {
                            cate: 8.0,
                        }
                    else:
                        importance_tuple = get_averaged_importance(
                            loaded_image,
                            # valid_objs,
                            'which part of this image evoke emotion?',
                            num_runs=1  # or whatever num_runs value you want
                        )

                    valid_objs = []
                    valid_masks = []
                    # importance_tuple['background'] = 10.00

                    for obj in list(importance_tuple.keys()):
                        # res_mask, _ = segment_emo_stimuli(shared_state.processed_image, obj)
                        res_mask = prompt2mask(loaded_image, obj)
                        if res_mask.max() > 0:
                            valid_objs.append(obj)
                            valid_masks.append(res_mask)

                    obj_list = []
                    score_list = []
                    for obj in valid_objs:
                        obj_list.append(obj)
                        score_list.append(importance_tuple[obj])

                    emotion_list = ['admirable', 'amusing', 'angry', 'annoying', 'approving', 'caring',
                                    'confusing', 'curious', 'desirable', 'disappointing', 'disapproving', 'disgusting',
                                    'embarrassing', 'exciting', 'grateful', 'grief', 'joyful', 'loving',
                                    'nervous', 'optimistic', 'proud', 'realization', 'relief', 'remorse',
                                    'sad', 'scary', 'surprising']

                    emotion_fashion_dict = {
                        'admirable': 'elegant, sophisticated, refined',
                        'amusing': 'playful, quirky, whimsical',
                        'angry': 'bold, edgy, aggressive',
                        'annoying': 'clashing, mismatched, chaotic',
                        'approving': 'polished, well-coordinated, harmonious',
                        'caring': 'comfortable, cozy, nurturing',
                        'confusing': 'avant-garde, experimental, unconventional',
                        'curious': 'eclectic, innovative, unique',
                        'desirable': 'luxurious, alluring, fashionable',
                        'disappointing': 'dull, unflattering, outdated',
                        'disapproving': 'conservative, restrictive, severe',
                        'disgusting': 'tacky, gaudy, excessive',
                        'embarrassing': 'ill-fitting, awkward, dated',
                        'exciting': 'dynamic, vibrant, statement-making',
                        'grateful': 'modest, balanced, appreciative',
                        'grief': 'dark, somber, muted',
                        'joyful': 'bright, flowing, uplifting',
                        'loving': 'romantic, soft, embracing',
                        'nervous': 'busy, fussy, over-detailed',
                        'optimistic': 'fresh, light, upbeat',
                        'proud': 'regal, structured, commanding',
                        'realization': 'transformative, revealing, eye-opening',
                        'relief': 'relaxed, fluid, easy',
                        'remorse': 'heavy, constrained, subdued',
                        'sad': 'drooping, loose, unstructured',
                        'scary': 'dramatic, intense, intimidating',
                        'surprising': 'unexpected, unconventional, innovative'
                    }
                    emotion_jewlery_dict = {
                        'admirable': 'timeless, masterful, exquisite',
                        'amusing': 'novelty, funky, playful',
                        'angry': 'sharp, angular, aggressive',
                        'annoying': 'noisy, cluttered, overwhelming',
                        'approving': 'classic, balanced, refined',
                        'caring': 'meaningful, personal, heartfelt',
                        'confusing': 'abstract, puzzling, complex',
                        'curious': 'intricate, detailed, fascinating',
                        'desirable': 'precious, coveted, stunning',
                        'disappointing': 'plain, uninspired, basic',
                        'disapproving': 'harsh, stern, rigid',
                        'disgusting': 'garish, over-the-top, tasteless',
                        'embarrassing': 'cheap-looking, juvenile, tacky',
                        'exciting': 'bold, eye-catching, dynamic',
                        'grateful': 'delicate, thoughtful, sincere',
                        'grief': 'heavy, dark, substantial',
                        'joyful': 'sparkling, lively, energetic',
                        'loving': 'romantic, soft, embracing',
                        'nervous': 'delicate, fragile, unstable',
                        'optimistic': 'light, airy, uplifting',
                        'proud': 'statement, grand, impressive',
                        'realization': 'clear, crystalline, revealing',
                        'relief': 'simple, clean, uncluttered',
                        'remorse': 'weighty, serious, substantial',
                        'sad': 'dull, tarnished, worn',
                        'scary': 'gothic, dark, mysterious',
                        'surprising': 'unusual, unexpected, distinctive'
                    }
                    emotion_vase_dict = {
                        'admirable': 'elegant, porcelain, classic',
                        'amusing': 'quirky, cartoon-like, whimsical',
                        'angry': 'aggressive, sharp, volcanic',
                        'annoying': 'noisy, cluttered, overwhelming',
                        'approving': 'classic, balanced, refined',
                        'caring': 'meaningful, personal, heartfelt',
                        'confusing': 'abstract, puzzling, complex',
                        'curious': 'intricate, detailed, fascinating',
                        'desirable': 'precious, coveted, stunning',
                        'disappointing': 'plain, uninspired, basic',
                        'disapproving': 'harsh, stern, rigid',
                        'disgusting': 'garish, over-the-top, tasteless',
                        'embarrassing': 'cheap-looking, juvenile, tacky',
                        'exciting': 'bold, eye-catching, dynamic',
                        'grateful': 'delicate, thoughtful, sincere',
                        'grief': 'heavy, dark, substantial',
                        'joyful': 'sparkling, lively, energetic',
                        'loving': 'romantic, soft, embracing',
                        'nervous': 'delicate, fragile, unstable',
                        'optimistic': 'light, airy, uplifting',
                        'proud': 'statement, grand, impressive',
                        'realization': 'clear, crystalline, revealing',
                        'relief': 'simple, clean, uncluttered',
                        'remorse': 'weighty, serious, substantial',
                        'sad': 'dull, tarnished, worn',
                        'scary': 'gothic, dark, mysterious',
                        'surprising': 'unusual, unexpected, distinctive'
                    }

                    for emotion in emotion_list:
                        if os.path.exists(f'output2_i4_{args.use_clip}/{cate}/{folder_name}/{emotion}{SEED}.png'):
                            continue
                        print(f"Processing emotion: {emotion}")
                        # Construct emotion question based on emotion

                        if cate == 'clothes' or cate == 'bag':
                            emotion_question = f"how could we make this {cate} look more {emotion} with {emotion_fashion_dict[emotion]}?"
                        elif cate == 'earring' or cate == 'necklace' or cate == 'ring' or cate == 'bracelet':
                            emotion_question = f"how could we make this {cate} look more {emotion} with {emotion_jewlery_dict[emotion]}?"
                        elif cate == 'vase':
                            emotion_question = f"how could we make this {cate} look more {emotion} with {emotion_vase_dict[emotion]}?"
                        else:
                            emotion_question = f"how could we make this {cate} look more {emotion}?"
                        # Get automatic prompts using LLaVA
                        # src_prompt = generate_src_prompts(loaded_image, 0)
                        tgt_prompt = generate_tgt_prompts(loaded_image,
                                                          emotion_question,
                                                          0)

                        # src_prompt = remove_repeating_sections(src_prompt)
                        tgt_prompt = remove_repeating_sections(tgt_prompt)

                        stimulus = extract_and_format_all(tgt_prompt)

                        img_desc_start = tgt_prompt.find("Image Description:")
                        img_desc_end = tgt_prompt.find("Overall Emotion:")
                        if img_desc_start != -1 and img_desc_end != -1:
                            # Keep only from Image Description onwards
                            tgt_prompt = tgt_prompt[img_desc_start:img_desc_end].replace("Image Description:", "").replace("Overall Emotion:", "")

                        tgt_prompt += stimulus
                        # src_prompt = clean_text(src_prompt)
                        tgt_prompt = clean_text(tgt_prompt)

                        # Perform the editing
                        edited_images, nsfw_detected = inference(img=Image.fromarray(loaded_image),
                                                  source_prompt='',
                                                  target_prompt=tgt_prompt,
                                                  local="",
                                                  mutual="",
                                                  positive_prompt="same age, masterpiece, best quality, high quality",
                                                  negative_prompt="different age, text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                                  guidance_s=0.6,
                                                  guidance_t=3.0 if (h >= 800 or w >= 800) else 2.0,
                                                  num_inference_steps=15,
                                                  width=512,
                                                  height=512,
                                                  seed=SEED,
                                                  strength=1.0,
                                                  cross_replace_steps=0.5 if (h >= 800 or w >= 800) else 0.7,
                                                  self_replace_steps=0.5 if (h >= 800 or w >= 800) else 0.7,
                                                  thresh_e=0.5,
                                                  thresh_m=0.5,
                                                  denoise=True,
                                                  emotion_stimulus_lst=valid_objs,
                                                  emotion_stimuli_masks=valid_masks,
                                                  importance_scores=score_list)
                        edited_images[-1][0].save(f'output2_i4_{args.use_clip}/{cate}/{folder_name}/{emotion}{SEED}.png')
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
