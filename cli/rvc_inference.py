#!/usr/bin/env python3
"""
AutoVid RVC Inference Script

A controlled wrapper around Mangio-RVC-Fork that properly handles all parameters
without modifying third-party code.
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count

# Add the Mangio-RVC-Fork directory to the Python path
mangio_rvc_path = Path(__file__).parent.parent / "third_party" / "Mangio-RVC-Fork"
sys.path.insert(0, str(mangio_rvc_path))

# Import RVC components
from vc_infer_pipeline import VC
from my_utils import load_audio, CSVutil
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

# Import the Config class from the RVC codebase
class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支援的N卡, 使用MPS进行推理")
            self.device = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            print("没有发现支援的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

    def use_fp32_config(self):
        for config_file in ["32k.json", "40k.json", "48k.json"]:
            with open(f"configs/{config_file}", "r") as f:
                strr = f.read().replace("true", "false")
            with open(f"configs/{config_file}", "w") as f:
                f.write(strr)


def setup_rvc_model(model_path, device, is_half):
    """Load and setup the RVC model."""
    print(f"Loading RVC model: {model_path}")
    
    # Load checkpoint
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    
    # Create config object and initialize VC pipeline
    config = Config(device, is_half)
    vc = VC(tgt_sr, config)
    
    # Load the model
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    
    # Load HuBERT model
    from fairseq import checkpoint_utils
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],  # File is in the RVC root directory
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()
    
    # Load generator based on version and if_f0
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    
    return vc, hubert_model, net_g, tgt_sr, if_f0, version, cpt


def convert_audio(
    model_path,
    index_path, 
    input_audio_path,
    output_audio_path,
    f0_up_key=0,
    f0_method="harvest",
    index_rate=0.66,
    device="cuda:0",
    is_half=True,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=1.0,
    protect=0.33,
    crepe_hop_length=128
):
    """
    Convert audio using RVC model.
    
    Args:
        model_path: Path to RVC model (.pth file)
        index_path: Path to index file (.index file)
        input_audio_path: Path to input audio file
        output_audio_path: Path to save converted audio
        f0_up_key: Pitch shift in semitones
        f0_method: F0 extraction method (harvest, pm, dio, crepe, mangio-crepe)
        index_rate: Index rate (0.0-1.0)
        device: Device to use (cuda:0, cpu, etc.)
        is_half: Use half precision
        filter_radius: Median filter radius
        resample_sr: Resample sample rate (0 for no resampling)
        rms_mix_rate: RMS mix rate
        protect: Protect voiceless consonants (0.0-0.5)
        crepe_hop_length: Hop length for CREPE method
    """
    try:
        # Setup model
        vc, hubert_model, net_g, tgt_sr, if_f0, version, cpt = setup_rvc_model(
            model_path, device, is_half
        )
        
        # Load and preprocess audio
        print(f"Loading audio: {input_audio_path}")
        audio = load_audio(input_audio_path, 16000, False, 0.0, 1.0)
        
        # Get speaker ID (assume single speaker model)
        sid = 0
        
        # Processing times (for benchmarking)
        times = [0, 0, 0]
        
        # Run inference
        print(f"Running RVC inference with method: {f0_method}")
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            index_path,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,  # This is the parameter that was missing
            f0_file=None,
        )
        
        # Save output
        print(f"Saving converted audio: {output_audio_path}")
        import soundfile as sf
        sf.write(output_audio_path, audio_opt, tgt_sr)
        
        print(f"Processing times: {times}")
        print("RVC conversion completed successfully!")
        return True
        
    except Exception as e:
        print(f"RVC conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="AutoVid RVC Inference")
    parser.add_argument("--model_path", required=True, help="Path to RVC model (.pth)")
    parser.add_argument("--index_path", required=True, help="Path to index file (.index)")
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output audio file")
    parser.add_argument("--f0_up_key", type=int, default=0, help="Pitch shift (semitones)")
    parser.add_argument("--f0_method", default="harvest", help="F0 extraction method")
    parser.add_argument("--index_rate", type=float, default=0.66, help="Index rate")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--is_half", action="store_true", help="Use half precision")
    parser.add_argument("--filter_radius", type=int, default=3, help="Median filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="Resample rate")
    parser.add_argument("--rms_mix_rate", type=float, default=1.0, help="RMS mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect consonants")
    parser.add_argument("--crepe_hop_length", type=int, default=128, help="CREPE hop length")
    
    args = parser.parse_args()
    
    success = convert_audio(
        model_path=args.model_path,
        index_path=args.index_path,
        input_audio_path=args.input,
        output_audio_path=args.output,
        f0_up_key=args.f0_up_key,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        device=args.device,
        is_half=args.is_half,
        filter_radius=args.filter_radius,
        resample_sr=args.resample_sr,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        crepe_hop_length=args.crepe_hop_length
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()