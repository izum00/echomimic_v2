import os
import datetime
import random
import traceback
import sys
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from moviepy.editor import VideoFileClip, AudioFileClip

import gradio as gr
from datetime import datetime
from torchao.quantization import quantize_, int8_weight_only
import gc

# ログ設定
def debug_log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

# エラーハンドリングデコレータ
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            debug_log(f"関数 {func.__name__} を開始します", "INFO")
            result = func(*args, **kwargs)
            debug_log(f"関数 {func.__name__} が正常に完了しました", "INFO")
            return result
        except Exception as e:
            error_msg = f"関数 {func.__name__} でエラーが発生: {str(e)}"
            debug_log(error_msg, "ERROR")
            debug_log(f"トレースバック:\n{traceback.format_exc()}", "ERROR")
            raise
    return wrapper

try:
    debug_log("システム情報を取得中...", "INFO")
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32mCUDAバージョン：{torch.version.cuda}\033[0m')
    print(f'\033[32mPytorchバージョン：{torch.__version__}\033[0m')
    print(f'\033[32mGPUモデル：{torch.cuda.get_device_name()}\033[0m')
    print(f'\033[32mVRAMサイズ：{total_vram_in_gb:.2f}GB\033[0m')
    print(f'\033[32m精度：float16\033[0m')
    debug_log("システム情報の取得が完了しました", "INFO")
except Exception as e:
    debug_log(f"システム情報の取得中にエラー: {e}", "ERROR")
    total_vram_in_gb = 0

dtype = torch.float16
if torch.cuda.is_available():
    device = "cuda"
    debug_log(f"CUDAデバイスを使用: {torch.cuda.get_device_name()}", "INFO")
else:
    print("cuda not available, using cpu")
    device = "cpu"
    debug_log("CUDAが利用不可のためCPUを使用します", "WARNING")

# FFmpeg設定
try:
    debug_log("FFmpeg設定を確認中...", "INFO")
    ffmpeg_path = os.getenv('FFMPEG_PATH')
    if ffmpeg_path is None:
        debug_log("FFMPEG_PATHが設定されていません", "WARNING")
        print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static")
    elif ffmpeg_path not in os.getenv('PATH'):
        debug_log(f"FFmpegをPATHに追加: {ffmpeg_path}", "INFO")
        os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"
    debug_log("FFmpeg設定の確認が完了しました", "INFO")
except Exception as e:
    debug_log(f"FFmpeg設定中にエラー: {e}", "ERROR")

# デフォルトパラメータ
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 768
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CFG = 1.0
DEFAULT_FPS = 24
DEFAULT_CONTEXT_FRAMES = 12
DEFAULT_CONTEXT_OVERLAP = 3

def validate_inputs(image_input, audio_input, pose_input, length):
    """入力パラメータの検証"""
    debug_log("入力パラメータの検証を開始", "INFO")
    
    if not image_input or not os.path.exists(image_input):
        raise ValueError(f"画像ファイルが見つかりません: {image_input}")
    
    if not audio_input or not os.path.exists(audio_input):
        raise ValueError(f"音声ファイルが見つかりません: {audio_input}")
    
    if not pose_input or not os.path.exists(pose_input):
        raise ValueError(f"ポーズディレクトリが見つかりません: {pose_input}")
    
    if not os.path.isdir(pose_input):
        raise ValueError(f"ポーズパスがディレクトリではありません: {pose_input}")
    
    # ポーズファイルの存在確認
    pose_files = [f for f in os.listdir(pose_input) if f.endswith('.npy')]
    if not pose_files:
        raise ValueError(f"ポーズディレクトリに.npyファイルが見つかりません: {pose_input}")
    
    debug_log(f"ポーズファイル数: {len(pose_files)}", "INFO")
    
    if length <= 0:
        raise ValueError(f"動画長は正の値である必要があります: {length}")
    
    debug_log("入力パラメータの検証が完了しました", "INFO")
    return True

@error_handler
def load_models(quantization_input):
    """モデルのロード"""
    debug_log("モデルロードを開始", "INFO")
    
    try:
        # VAE
        debug_log("VAEモデルをロード中...", "INFO")
        vae = AutoencoderKL.from_pretrained("./pretrained_weights/sd-vae-ft-mse").to(device, dtype=dtype)
        if quantization_input:
            debug_log("VAEにint8量子化を適用", "INFO")
            quantize_(vae, int8_weight_only())
            print("int8量化")
        debug_log("VAEモデルのロードが完了しました", "INFO")
        
        # Reference UNet
        debug_log("Reference UNetモデルをロード中...", "INFO")
        reference_unet = UNet2DConditionModel.from_pretrained(
            "./pretrained_weights/sd-image-variations-diffusers", 
            subfolder="unet", 
            use_safetensors=False
        ).to(dtype=dtype, device=device)
        reference_unet.load_state_dict(torch.load("./pretrained_weights/reference_unet.pth", weights_only=True))
        if quantization_input:
            debug_log("Reference UNetにint8量子化を適用", "INFO")
            quantize_(reference_unet, int8_weight_only())
        debug_log("Reference UNetモデルのロードが完了しました", "INFO")
        
        # Denoising UNet
        debug_log("Denoising UNetモデルをロード中...", "INFO")
        if os.path.exists("./pretrained_weights/motion_module_acc.pth"):
            debug_log("motion_module_acc.pthが見つかりました", "INFO")
        else:
            raise FileNotFoundError("motion_module_acc.pthが見つかりません")
        
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            "./pretrained_weights/sd-image-variations-diffusers",
            "./pretrained_weights/motion_module_acc.pth",
            subfolder="unet",
            unet_additional_kwargs={
                "use_inflated_groupnorm": True,
                "unet_use_cross_frame_attention": False,
                "unet_use_temporal_attention": False,
                "use_motion_module": True,
                "cross_attention_dim": 384,
                "motion_module_resolutions": [1, 2, 4, 8],
                "motion_module_mid_block": True,
                "motion_module_decoder_only": False,
                "motion_module_type": "Vanilla",
                "motion_module_kwargs": {
                    "num_attention_heads": 8,
                    "num_transformer_block": 1,
                    "attention_block_types": ['Temporal_Self', 'Temporal_Self'],
                    "temporal_position_encoding": True,
                    "temporal_position_encoding_max_len": 32,
                    "temporal_attention_dim_div": 1,
                }
            },
        ).to(dtype=dtype, device=device)
        denoising_unet.load_state_dict(
            torch.load("./pretrained_weights/denoising_unet_acc.pth", weights_only=True),
            strict=False
        )
        debug_log("Denoising UNetモデルのロードが完了しました", "INFO")
        
        # Pose Net
        debug_log("Pose Netモデルをロード中...", "INFO")
        pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(dtype=dtype, device=device)
        pose_net.load_state_dict(torch.load("./pretrained_weights/pose_encoder.pth", weights_only=True))
        debug_log("Pose Netモデルのロードが完了しました", "INFO")
        
        # Audio Processor
        debug_log("Audio Processorモデルをロード中...", "INFO")
        audio_processor = load_audio_model(model_path="./pretrained_weights/audio_processor/tiny.pt", device=device)
        debug_log("Audio Processorモデルのロードが完了しました", "INFO")
        
        debug_log("すべてのモデルのロードが完了しました", "INFO")
        return vae, reference_unet, denoising_unet, pose_net, audio_processor
        
    except Exception as e:
        debug_log(f"モデルロード中にエラーが発生: {e}", "ERROR")
        raise

@error_handler
def generate(
    image_input, 
    audio_input, 
    pose_input, 
    width, 
    height, 
    length, 
    steps, 
    sample_rate, 
    cfg, 
    fps, 
    context_frames, 
    context_overlap, 
    quantization_input, 
    seed
):
    debug_log("===== 動画生成処理を開始 =====", "INFO")
    debug_log(f"入力パラメータ: 画像={image_input}, 音声={audio_input}, ポーズ={pose_input}", "INFO")
    debug_log(f"生成パラメータ: 長さ={length}, ステップ={steps}, FPS={fps}, 量子化={quantization_input}", "INFO")
    debug_log(f"コンテキスト: フレーム={context_frames}, オーバーラップ={context_overlap}", "INFO")
    
    # メモリクリーン
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    debug_log("メモリクリーン完了", "INFO")
    
    # タイムスタンプと出力ディレクトリ
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("outputs")
    save_dir.mkdir(exist_ok=True, parents=True)
    save_name = f"{save_dir}/{timestamp}"
    debug_log(f"出力ディレクトリ: {save_dir}", "INFO")
    
    try:
        # 入力検証
        validate_inputs(image_input, audio_input, pose_input, length)
        
        # モデルロード
        debug_log("モデルロードを開始...", "INFO")
        vae, reference_unet, denoising_unet, pose_net, audio_processor = load_models(quantization_input)
        debug_log("モデルロード完了", "INFO")
        
        # スケジューラ設定
        debug_log("スケジューラ設定中...", "INFO")
        sched_kwargs = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "clip_sample": False,
            "steps_offset": 1,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "timestep_spacing": "trailing"
        }
        scheduler = DDIMScheduler(**sched_kwargs)
        debug_log("スケジューラ設定完了", "INFO")
        
        # パイプライン作成
        debug_log("パイプラインを作成中...", "INFO")
        pipe = EchoMimicV2Pipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            audio_guider=audio_processor,
            pose_encoder=pose_net,
            scheduler=scheduler,
        )
        pipe = pipe.to(device, dtype=dtype)
        debug_log("パイプライン作成完了", "INFO")
        
        # シード設定
        if seed is not None and seed > -1:
            generator = torch.manual_seed(seed)
            debug_log(f"シードを使用: {seed}", "INFO")
        else:
            seed = random.randint(100, 1000000)
            generator = torch.manual_seed(seed)
            debug_log(f"ランダムシード生成: {seed}", "INFO")
        
        # 画像読み込み
        debug_log(f"リファレンス画像読み込み中: {image_input}", "INFO")
        ref_image_pil = Image.open(image_input).resize((width, height))
        
        # 音声読み込み
        debug_log(f"音声ファイル読み込み中: {audio_input}", "INFO")
        audio_clip = AudioFileClip(audio_input)
        audio_duration = audio_clip.duration
        debug_log(f"音声長さ: {audio_duration}秒", "INFO")
        
        # 動画長の調整
        max_length = int(audio_duration * fps)
        pose_files = len([f for f in os.listdir(pose_input) if f.endswith('.npy')])
        max_length = min(max_length, pose_files)
        length = min(length, max_length)
        debug_log(f"動画長を {length} フレームに調整しました", "INFO")
        
        # ポーズデータ読み込み
        debug_log("ポーズデータを読み込み中...", "INFO")
        start_idx = 0
        pose_list = []
        for index in range(start_idx, start_idx + length):
            debug_log(f"ポーズフレーム {index} を読み込み中...", "DEBUG")
            tgt_musk = np.zeros((width, height, 3)).astype('uint8')
            tgt_musk_path = os.path.join(pose_input, f"{index}.npy")
            
            if not os.path.exists(tgt_musk_path):
                debug_log(f"ポーズファイルが見つかりません: {tgt_musk_path}", "WARNING")
                continue
                
            detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
            im = np.transpose(np.array(im), (1, 2, 0))
            tgt_musk[rb:re, cb:ce, :] = im
            
            tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
            pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=dtype, device=device).permute(2, 0, 1) / 255.0)
        
        if not pose_list:
            raise ValueError("有効なポーズデータが読み込めませんでした")
        
        poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        debug_log(f"ポーズデータ読み込み完了: {len(pose_list)}フレーム", "INFO")
        
        # 動画生成
        debug_log("動画生成を開始...", "INFO")
        debug_log(f"生成パラメータ: 幅={width}, 高さ={height}, フレーム数={length}", "INFO")
        
        video = pipe(
            ref_image_pil,
            audio_input,
            poses_tensor[:, :, :length, ...],
            width,
            height,
            length,
            steps,
            cfg,
            generator=generator,
            audio_sample_rate=sample_rate,
            context_frames=context_frames,
            fps=fps,
            context_overlap=context_overlap,
            start_idx=start_idx,
        ).videos
        
        debug_log(f"動画生成完了: 形状={video.shape}", "INFO")
        
        # 動画保存
        debug_log("動画を保存中...", "INFO")
        final_length = min(video.shape[2], poses_tensor.shape[2], length)
        video_sig = video[:, :, :final_length, :, :]
        
        save_videos_grid(
            video_sig,
            save_name + "_woa_sig.mp4",
            n_rows=1,
            fps=fps,
        )
        debug_log("音声なし動画を保存しました", "INFO")
        
        # 音声合成
        debug_log("音声を合成中...", "INFO")
        video_clip_sig = VideoFileClip(save_name + "_woa_sig.mp4")
        video_clip_sig = video_clip_sig.set_audio(audio_clip)
        video_clip_sig.write_videofile(save_name + "_sig.mp4", codec="libx264", audio_codec="aac", threads=2)
        debug_log(f"音声合成完了: {save_name}_sig.mp4", "INFO")
        
        video_output = save_name + "_sig.mp4"
        seed_text = gr.update(visible=True, value=seed)
        
        # メモリクリーン
        del pipe, vae, reference_unet, denoising_unet, pose_net, audio_processor
        gc.collect()
        torch.cuda.empty_cache()
        debug_log("メモリをクリーンアップしました", "INFO")
        
        debug_log("===== 動画生成処理が正常に完了しました =====", "INFO")
        return video_output, seed_text
        
    except Exception as e:
        debug_log(f"動画生成中にエラーが発生: {e}", "ERROR")
        debug_log(f"エラー詳細:\n{traceback.format_exc()}", "ERROR")
        raise

# Gradioインターフェース
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">EchoMimicV2-ACC</h2>
            </div>
            <div style="text-align: center;">
                <a href="https://github.com/antgroup/echomimic_v2">🌐 Github</a> |
                <a href="https://arxiv.org/abs/2411.10061">📜 arXiv </a>
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 該当デモは学術研究および体験のみを目的としています
            </div>
            """)
    
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image_input = gr.Image(label="画像入力（自動スケーリング）", type="filepath")
                    audio_input = gr.Audio(label="音声入力", type="filepath")
                    pose_input = gr.Textbox(
                        label="ポーズ入力（ディレクトリパス）", 
                        placeholder="ポーズデータのディレクトリパスを入力", 
                        value="assets/halfbody_demo/pose/fight"
                    )
                with gr.Group():
                    # スライダーと数値入力用のコンポーネント
                    width = gr.Slider(
                        minimum=256, maximum=1024, value=DEFAULT_WIDTH, step=64,
                        label="幅 (width)"
                    )
                    height = gr.Slider(
                        minimum=256, maximum=1024, value=DEFAULT_HEIGHT, step=64,
                        label="高さ (height)"
                    )
                    length = gr.Slider(
                        minimum=1, maximum=500, value=120, step=1,
                        label="動画長（フレーム数）"
                    )
                    steps = gr.Slider(
                        minimum=1, maximum=50, value=6, step=1,
                        label="推論ステップ数"
                    )
                    sample_rate = gr.Number(
                        value=DEFAULT_SAMPLE_RATE, label="サンプリングレート", visible=False
                    )
                    cfg = gr.Number(
                        value=DEFAULT_CFG, label="CFGスケール", visible=False
                    )
                    fps = gr.Number(
                        value=DEFAULT_FPS, label="FPS", visible=False
                    )
                    context_frames = gr.Number(
                        value=DEFAULT_CONTEXT_FRAMES, label="コンテキストフレーム数", visible=False
                    )
                    context_overlap = gr.Number(
                        value=DEFAULT_CONTEXT_OVERLAP, label="コンテキストオーバーラップ", visible=False
                    )
                    quantization_input = gr.Checkbox(
                        label="int8量子化（VRAM 12GBのユーザーに推奨、5秒以内の音声を使用）", 
                        value=False
                    )
                    seed = gr.Number(
                        label="シード（-1はランダム）", value=-1
                    )
                generate_button = gr.Button("🎬 動画生成")
            with gr.Column():
                video_output = gr.Video(label="出力動画")
                seed_text = gr.Textbox(label="シード", interactive=False, visible=False)
        
        gr.Examples(
            examples=[
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0003.png", "assets/halfbody_demo/audio/chinese/fighting.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0033.png", "assets/halfbody_demo/audio/chinese/good.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0010.png", "assets/halfbody_demo/audio/chinese/news.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/1168.png", "assets/halfbody_demo/audio/chinese/no_smoking.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0057.png", "assets/halfbody_demo/audio/chinese/ultraman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0001.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0077.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
            ],
            inputs=[image_input, audio_input],
            label="プリセット画像と音声",
        )
    
    def generate_with_logging(*args, **kwargs):
        try:
            debug_log("Gradioからの生成リクエストを受信", "INFO")
            debug_log(f"受信した引数の数: {len(args)}", "DEBUG")
            result = generate(*args, **kwargs)
            debug_log("生成リクエストが正常に完了", "INFO")
            return result
        except Exception as e:
            debug_log(f"生成リクエストでエラー: {e}", "ERROR")
            error_msg = f"エラーが発生しました: {str(e)}"
            return None, gr.update(visible=True, value=f"エラー: {str(e)}")
    
    # 修正: すべての入力はコンポーネント参照を使用
    generate_button.click(
        generate_with_logging,
        inputs=[
            image_input,      # gr.Image
            audio_input,      # gr.Audio
            pose_input,       # gr.Textbox
            width,           # gr.Slider
            height,          # gr.Slider
            length,          # gr.Slider
            steps,           # gr.Slider
            sample_rate,     # gr.Number (hidden)
            cfg,             # gr.Number (hidden)
            fps,             # gr.Number (hidden)
            context_frames,  # gr.Number (hidden)
            context_overlap, # gr.Number (hidden)
            quantization_input, # gr.Checkbox
            seed             # gr.Number
        ],
        outputs=[video_output, seed_text],
    )

if __name__ == "__main__":
    debug_log("Gradioアプリケーションを起動", "INFO")
    demo.queue()
    demo.launch(inbrowser=True, share=True, debug=True)
    debug_log("Gradioアプリケーションを終了", "INFO")
