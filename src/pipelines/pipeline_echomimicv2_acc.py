import inspect
import math
import logging
import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm

from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler
from src.pipelines.utils import get_tensor_interpolation_method
from src.pipelines.step_func import origin_by_velocity_and_sample, psuedo_velocity_wrt_noisy_and_timestep

# ロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def debug_log(msg, level="INFO"):
    """デバッグログ出力関数"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if level == "ERROR":
        logger.error(f"[{timestamp}] {msg}")
    elif level == "WARNING":
        logger.warning(f"[{timestamp}] {msg}")
    else:
        logger.info(f"[{timestamp}] {msg}")
    print(f"[{timestamp}] [{level}] {msg}")

def error_handler(func):
    """エラーハンドリングデコレータ"""
    def wrapper(*args, **kwargs):
        try:
            debug_log(f"関数 {func.__name__} を開始", "INFO")
            result = func(*args, **kwargs)
            debug_log(f"関数 {func.__name__} が正常に完了", "INFO")
            return result
        except Exception as e:
            debug_log(f"関数 {func.__name__} でエラー: {str(e)}", "ERROR")
            debug_log(f"トレースバック:\n{traceback.format_exc()}", "ERROR")
            raise
    return wrapper

@dataclass
class EchoMimicV2PipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class EchoMimicV2Pipeline(DiffusionPipeline):

    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        audio_guider,
        pose_encoder,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
    ):
        debug_log("EchoMimicV2Pipelineの初期化を開始", "INFO")
        try:
            super().__init__()

            self.register_modules(
                vae=vae,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                audio_guider=audio_guider,
                pose_encoder=pose_encoder,
                scheduler=scheduler,
                image_proj_model=image_proj_model,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )
            
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.ref_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
            )
            
            # パラメータの検証
            if self.vae is None:
                raise ValueError("VAEが初期化されていません")
            if self.reference_unet is None:
                raise ValueError("Reference UNetが初期化されていません")
            if self.denoising_unet is None:
                raise ValueError("Denoising UNetが初期化されていません")
            if self.audio_guider is None:
                raise ValueError("Audio Guiderが初期化されていません")
            if self.pose_encoder is None:
                raise ValueError("Pose Encoderが初期化されていません")
            if self.scheduler is None:
                raise ValueError("Schedulerが初期化されていません")
                
            debug_log("EchoMimicV2Pipelineの初期化が完了しました", "INFO")
        except Exception as e:
            debug_log(f"パイプライン初期化中にエラー: {str(e)}", "ERROR")
            raise

    def enable_vae_slicing(self):
        try:
            self.vae.enable_slicing()
            debug_log("VAEスライシングを有効化しました", "INFO")
        except Exception as e:
            debug_log(f"VAEスライシング有効化中にエラー: {str(e)}", "WARNING")
            raise

    def disable_vae_slicing(self):
        try:
            self.vae.disable_slicing()
            debug_log("VAEスライシングを無効化しました", "INFO")
        except Exception as e:
            debug_log(f"VAEスライシング無効化中にエラー: {str(e)}", "WARNING")
            raise

    def enable_sequential_cpu_offload(self, gpu_id=0):
        try:
            if is_accelerate_available():
                from accelerate import cpu_offload
            else:
                raise ImportError("Please install accelerate via `pip install accelerate`")

            device = torch.device(f"cuda:{gpu_id}")

            for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
                if cpu_offloaded_model is not None:
                    cpu_offload(cpu_offloaded_model, device)
            debug_log(f"シーケンシャルCPUオフロードを有効化しました (GPU ID: {gpu_id})", "INFO")
        except Exception as e:
            debug_log(f"CPUオフロード有効化中にエラー: {str(e)}", "ERROR")
            raise

    @property
    def _execution_device(self):
        try:
            if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
                return self.device
            for module in self.unet.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
            return self.device
        except Exception as e:
            debug_log(f"_execution_device取得中にエラー: {str(e)}", "ERROR")
            raise

    def decode_latents(self, latents):
        debug_log(f"潜在変数のデコードを開始: 形状={latents.shape}", "INFO")
        try:
            video_length = latents.shape[2]
            latents = 1 / 0.18215 * latents
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            video = []
            
            for frame_idx in tqdm(range(latents.shape[0]), desc="デコード中"):
                try:
                    video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
                except Exception as e:
                    debug_log(f"フレーム {frame_idx} のデコード中にエラー: {str(e)}", "WARNING")
                    raise
            
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.cpu().float().numpy()
            
            debug_log(f"デコード完了: 形状={video.shape}", "INFO")
            return video
        except Exception as e:
            debug_log(f"潜在変数のデコード中にエラー: {str(e)}", "ERROR")
            raise

    def prepare_extra_step_kwargs(self, generator, eta):
        debug_log(f"ステップ用の追加引数を準備中: eta={eta}", "DEBUG")
        try:
            accepts_eta = "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            accepts_generator = "generator" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
                
            debug_log("追加引数の準備が完了しました", "DEBUG")
            return extra_step_kwargs
        except Exception as e:
            debug_log(f"追加引数準備中にエラー: {str(e)}", "ERROR")
            raise

    def prepare_latents_bp(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        debug_log(f"潜在変数を準備中 (BP): サイズ={batch_size}, チャネル={num_channels_latents}", "INFO")
        try:
            shape = (
                batch_size,
                num_channels_latents,
                video_length,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"ジェネレータのリスト長 {len(generator)} がバッチサイズ {batch_size} と一致しません"
                )

            if latents is None:
                latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
            else:
                latents = latents.to(device)

            latents = latents * self.scheduler.init_noise_sigma
            debug_log(f"潜在変数の準備完了: 形状={latents.shape}", "INFO")
            return latents
        except Exception as e:
            debug_log(f"潜在変数準備中にエラー: {str(e)}", "ERROR")
            raise

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        context_frame_length
    ):
        debug_log(f"潜在変数を準備中: サイズ={batch_size}, チャネル={num_channels_latents}, コンテキストフレーム={context_frame_length}", "INFO")
        try:
            shape = (
                batch_size,
                num_channels_latents,
                video_length,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"ジェネレータのリスト長 {len(generator)} がバッチサイズ {batch_size} と一致しません"
                )

            latents_seg = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
            latents = latents_seg
            
            latents = latents * self.scheduler.init_noise_sigma
            debug_log(f"潜在変数の準備完了: 形状={latents.shape}", "INFO")
            return latents
        except Exception as e:
            debug_log(f"潜在変数準備中にエラー: {str(e)}", "ERROR")
            raise

    def prepare_latents_smooth(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        context_frame_length
    ):
        debug_log(f"スムース潜在変数を準備中: サイズ={batch_size}, チャネル={num_channels_latents}", "INFO")
        try:
            shape = (
                batch_size,
                num_channels_latents,
                video_length,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"ジェネレータのリスト長 {len(generator)} がバッチサイズ {batch_size} と一致しません"
                )

            latents_seg = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
            latents = latents_seg
            latents = torch.clamp(latents_seg, -1.5, 1.5)
            latents = latents * self.scheduler.init_noise_sigma
            
            debug_log(f"スムース潜在変数の準備完了: 形状={latents.shape}", "INFO")
            return latents
        except Exception as e:
            debug_log(f"スムース潜在変数準備中にエラー: {str(e)}", "ERROR")
            raise

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        debug_log(f"潜在変数の補間を開始: 補間係数={interpolation_factor}", "INFO")
        try:
            if interpolation_factor < 2:
                debug_log("補間係数が2未満のためスキップ", "DEBUG")
                return latents

            new_latents = torch.zeros(
                (
                    latents.shape[0],
                    latents.shape[1],
                    ((latents.shape[2] - 1) * interpolation_factor) + 1,
                    latents.shape[3],
                    latents.shape[4],
                ),
                device=latents.device,
                dtype=latents.dtype,
            )

            org_video_length = latents.shape[2]
            rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

            new_index = 0
            v0 = None
            v1 = None

            for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
                v0 = latents[:, :, i0, :, :]
                v1 = latents[:, :, i1, :, :]

                new_latents[:, :, new_index, :, :] = v0
                new_index += 1

                for f in rate:
                    try:
                        v = get_tensor_interpolation_method()(
                            v0.to(device=device), v1.to(device=device), f
                        )
                        new_latents[:, :, new_index, :, :] = v.to(latents.device)
                        new_index += 1
                    except Exception as e:
                        debug_log(f"補間中にエラー (インデックス: {i0}->{i1}, f={f}): {str(e)}", "WARNING")
                        raise

            new_latents[:, :, new_index, :, :] = v1
            new_index += 1

            debug_log(f"補間完了: 形状={new_latents.shape}", "INFO")
            return new_latents
        except Exception as e:
            debug_log(f"潜在変数補間中にエラー: {str(e)}", "ERROR")
            raise

    @torch.no_grad()
    @error_handler
    def __call__(
        self,
        ref_image,
        audio_path,
        poses_tensor,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=12,
        context_stride=1,
        context_overlap=0,
        context_batch_size=1,
        interpolation_factor=1,
        audio_sample_rate=16000,
        fps=25,
        audio_margin=2,
        start_idx=0,
        **kwargs,
    ):
        debug_log("===== EchoMimicV2Pipeline __call__ 開始 =====", "INFO")
        debug_log(f"入力パラメータ: 幅={width}, 高さ={height}, 動画長={video_length}, ステップ={num_inference_steps}", "INFO")
        debug_log(f"コンテキスト: フレーム={context_frames}, ストライド={context_stride}, オーバーラップ={context_overlap}", "INFO")
        
        try:
            # 入力パラメータの検証
            if ref_image is None:
                raise ValueError("参照画像が指定されていません")
            if audio_path is None:
                raise ValueError("音声パスが指定されていません")
            if poses_tensor is None:
                raise ValueError("ポーズテンソルが指定されていません")
            if num_inference_steps <= 0:
                raise ValueError(f"推論ステップ数は正の値である必要があります: {num_inference_steps}")
            if guidance_scale < 1.0:
                debug_log(f"ガイダンススケールが1.0未満です: {guidance_scale}", "WARNING")
            if video_length <= 0:
                raise ValueError(f"動画長は正の値である必要があります: {video_length}")
                
            # デフォルト値の設定
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            
            device = self._execution_device
            debug_log(f"実行デバイス: {device}", "INFO")
            
            do_classifier_free_guidance = guidance_scale > 1.0
            debug_log(f"分類器フリーガイダンス: {do_classifier_free_guidance}", "DEBUG")

            # タイムステップの準備
            debug_log("タイムステップを準備中...", "INFO")
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            debug_log(f"タイムステップ数: {len(timesteps)}", "INFO")

            batch_size = 1

            # 参照制御の設定
            debug_log("参照制御を設定中...", "INFO")
            try:
                reference_control_writer = ReferenceAttentionControl(
                    self.reference_unet,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    mode="write",
                    batch_size=batch_size,
                    fusion_blocks="full",
                )
                reference_control_reader = ReferenceAttentionControl(
                    self.denoising_unet,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    mode="read",
                    batch_size=batch_size,
                    fusion_blocks="full",
                )
                debug_log("参照制御の設定が完了しました", "INFO")
            except Exception as e:
                debug_log(f"参照制御の設定中にエラー: {str(e)}", "ERROR")
                raise

            # 音声特徴量の抽出
            debug_log(f"音声特徴量を抽出中: {audio_path}", "INFO")
            try:
                whisper_feature = self.audio_guider.audio2feat(audio_path)
                whisper_chunks = self.audio_guider.feature2chunks(
                    feature_array=whisper_feature, fps=fps
                )
                audio_frame_num = whisper_chunks.shape[0]
                audio_fea_final = torch.Tensor(whisper_chunks).to(
                    dtype=self.vae.dtype, device=self.vae.device
                )
                audio_fea_final = audio_fea_final.unsqueeze(0)
                debug_log(f"音声特徴量抽出完了: フレーム数={audio_frame_num}, 形状={audio_fea_final.shape}", "INFO")
            except Exception as e:
                debug_log(f"音声特徴量抽出中にエラー: {str(e)}", "ERROR")
                raise

            # 動画長の調整
            video_length = min(video_length, audio_frame_num)
            debug_log(f"調整後の動画長: {video_length}", "INFO")

            # 潜在変数の準備
            num_channels_latents = self.denoising_unet.in_channels
            debug_log(f"潜在変数を準備中: チャネル数={num_channels_latents}", "INFO")
            latents = self.prepare_latents_smooth(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                width,
                height,
                video_length,
                audio_fea_final.dtype,
                device,
                generator,
                context_frames
            )
            debug_log(f"潜在変数準備完了: 形状={latents.shape}", "INFO")

            # ポーズエンコーダ
            debug_log("ポーズエンコーダを適用中...", "INFO")
            try:
                pose_enocder_tensor = self.pose_encoder(poses_tensor)
                debug_log(f"ポーズエンコーダ完了: 形状={pose_enocder_tensor.shape}", "INFO")
            except Exception as e:
                debug_log(f"ポーズエンコーダ適用中にエラー: {str(e)}", "ERROR")
                raise

            # 追加引数
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 参照画像の潜在変数
            debug_log("参照画像の潜在変数を準備中...", "INFO")
            try:
                ref_image_tensor = self.ref_image_processor.preprocess(
                    ref_image, height=height, width=width
                )
                ref_image_tensor = ref_image_tensor.to(
                    dtype=self.vae.dtype, device=self.vae.device
                )
                ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
                ref_image_latents = ref_image_latents * 0.18215
                debug_log(f"参照画像潜在変数準備完了: 形状={ref_image_latents.shape}", "INFO")
            except Exception as e:
                debug_log(f"参照画像処理中にエラー: {str(e)}", "ERROR")
                raise

            # コンテキストスケジューラ
            context_scheduler = get_context_scheduler(context_schedule)
            context_queue = list(
                context_scheduler(
                    0,
                    num_inference_steps,
                    latents.shape[2],
                    context_frames,
                    context_stride,
                    context_overlap,
                )
            )
            debug_log(f"コンテキストキュー作成完了: サイズ={len(context_queue)}", "INFO")

            # デノイジングループ
            debug_log("デノイジングループを開始", "INFO")
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for t_i, t in enumerate(timesteps):
                    debug_log(f"ステップ {t_i+1}/{num_inference_steps}, タイムスタンプ={t}", "DEBUG")
                    
                    try:
                        noise_pred = torch.zeros(
                            (
                                latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                                *latents.shape[1:],
                            ),
                            device=latents.device,
                            dtype=latents.dtype,
                        )
                        counter = torch.zeros(
                            (1, 1, latents.shape[2], 1, 1),
                            device=latents.device,
                            dtype=latents.dtype,
                        )

                        # 1. Forward reference image
                        if t_i == 0:
                            debug_log("参照画像をフォワード伝搬中...", "INFO")
                            self.reference_unet(
                                ref_image_latents,
                                torch.zeros_like(t),
                                encoder_hidden_states=None,
                                return_dict=False,
                            )
                            reference_control_reader.update(
                                reference_control_writer, 
                                do_classifier_free_guidance=do_classifier_free_guidance
                            )
                            debug_log("参照画像フォワード伝搬完了", "INFO")

                        num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                        global_context = []
                        for j in range(num_context_batches):
                            global_context.append(
                                context_queue[
                                    j * context_batch_size : (j + 1) * context_batch_size
                                ]
                            )

                        ## refine
                        for context_idx, context in enumerate(global_context):
                            debug_log(f"コンテキストバッチ {context_idx+1}/{num_context_batches} を処理中", "DEBUG")
                            
                            new_context = [[0 for _ in range(len(context[c_j]))] for c_j in range(len(context))]
                            for c_j in range(len(context)):
                                for c_i in range(len(context[c_j])):
                                    new_context[c_j][c_i] = (context[c_j][c_i] + t_i * 3) % video_length

                            latent_model_input = (
                                torch.cat([latents[:, :, c] for c in new_context])
                                .to(device)
                                .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                            )

                            audio_latents_cond = torch.cat([audio_fea_final[:, c] for c in new_context]).to(device)
                            audio_latents = torch.cat([torch.zeros_like(audio_latents_cond), audio_latents_cond], 0)
                            
                            pose_latents_cond = torch.cat([pose_enocder_tensor[:, :, c] for c in new_context]).to(device)
                            pose_latents = torch.cat([torch.zeros_like(pose_latents_cond), pose_latents_cond], 0)
                            
                            latent_model_input = self.scheduler.scale_model_input(
                                latent_model_input, t
                            )
                            b, c, f, h, w = latent_model_input.shape
                            
                            # UNet予測
                            try:
                                pred = self.denoising_unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=None,
                                    audio_cond_fea=audio_latents if do_classifier_free_guidance else audio_latents_cond,
                                    face_musk_fea=pose_latents if do_classifier_free_guidance else pose_latents_cond,
                                    return_dict=False,
                                )[0]
                            except Exception as e:
                                debug_log(f"UNet予測中にエラー (コンテキスト {context_idx}): {str(e)}", "ERROR")
                                raise

                            # 予測の更新
                            alphas_cumprod = self.scheduler.alphas_cumprod.to(latent_model_input.device)
                            x_pred = origin_by_velocity_and_sample(pred, latent_model_input, alphas_cumprod, t)
                            pred = psuedo_velocity_wrt_noisy_and_timestep(
                                latent_model_input, x_pred, alphas_cumprod, t, torch.ones_like(t) * (-1)
                            )

                            for j, c in enumerate(new_context):
                                noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                                counter[:, :, c] = counter[:, :, c] + 1

                        # ガイダンス
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )
                        else:
                            noise_pred = noise_pred / counter
                            
                        latents = self.scheduler.step(
                            noise_pred, t, latents, **extra_step_kwargs
                        ).prev_sample

                        if t_i == len(timesteps) - 1 or (
                            (t_i + 1) > num_warmup_steps and (t_i + 1) % self.scheduler.order == 0
                        ):
                            progress_bar.update()
                            
                    except Exception as e:
                        debug_log(f"デノイジングステップ {t_i} でエラー: {str(e)}", "ERROR")
                        raise

            # クリーンアップ
            debug_log("参照制御をクリア中...", "INFO")
            reference_control_reader.clear()
            reference_control_writer.clear()
            debug_log("参照制御をクリアしました", "INFO")

            # 補間
            if interpolation_factor > 0:
                debug_log(f"潜在変数を補間中: 係数={interpolation_factor}", "INFO")
                latents = self.interpolate_latents(latents, interpolation_factor, device)
                debug_log(f"補間完了: 形状={latents.shape}", "INFO")

            # ポスト処理
            debug_log("デコードを開始...", "INFO")
            images = self.decode_latents(latents)
            debug_log(f"デコード完了: 形状={images.shape}", "INFO")

            # テンソル変換
            if output_type == "tensor":
                images = torch.from_numpy(images)

            debug_log("===== EchoMimicV2Pipeline __call__ 完了 =====", "INFO")
            
            if not return_dict:
                return images

            return EchoMimicV2PipelineOutput(videos=images)
            
        except Exception as e:
            debug_log(f"パイプライン実行中にエラー: {str(e)}", "ERROR")
            debug_log(f"トレースバック:\n{traceback.format_exc()}", "ERROR")
            raise
