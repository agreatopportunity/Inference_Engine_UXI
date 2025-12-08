"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SOVRA OMNI v3.0 - Neural Interface for LLM Inference                         ║
║  ─────────────────────────────────────────────────────────────────────────────║
║  Supports:                                                                    ║
║    • Native Models (.pt) - Custom LLaMA-3 architecture                        ║
║    • GGUF Models (.gguf) - DeepSeek, Llama, Mistral, Qwen3, etc.             ║
║    • Voice Input/Output - Speech-to-text and TTS                              ║
║    • Training Interface - Fine-tune models                                    ║
║                                                                               ║
║  Features:                                                                    ║
║    • Mobile & Desktop responsive design                                       ║
║    • Dark/Light theme toggle                                                  ║
║    • Real-time streaming generation                                           ║
║    • GPU memory monitoring                                                    ║
║    • Multi-GPU support                                                        ║
║    • HuggingFace model downloader                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
import contextlib
import argparse
import os
import sys
import time
import json
import threading
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime

# Optional: huggingface_hub for better downloads
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️  huggingface_hub not installed. Using direct downloads.")

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SOVRA OMNI v3.0 - Command Line Arguments                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  --device_id  : GPU Index to use (Default: 0)                                 ║
║  --port       : Port to run the UI on (Default: 7860)                         ║
║  --share      : Create a public Gradio share link                             ║
║                                                                               ║
║  Example: python3 app.py --device_id 0 --port 7860 --share                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """)
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    ARGS_DEVICE = args.device_id
else:
    ARGS_DEVICE = 0
    args = type('obj', (object,), {'port': 7860, 'share': False})()

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from model_llama3 import GPT, GPTConfig
    NATIVE_AVAILABLE = True
except ImportError:
    print("⚠️  model_llama3.py not found. Native models disabled.")
    NATIVE_AVAILABLE = False

try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    print("⚠️  llama-cpp-python not installed. GGUF models disabled.")
    GGUF_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════
CURRENT_MODEL = None
CURRENT_ENGINE = None
MODEL_INFO = {}
ENC = None
STOP_GENERATION = False
CURRENT_TEMPLATE = "None (Raw)"
DOWNLOAD_PROGRESS = {"status": "", "progress": 0}
VOICE_ENABLED = False
TRAINING_ACTIVE = False

# Models directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CATALOG - December 2025 (Optimized for 12GB VRAM)
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_CATALOG = {
    "🔥 Qwen3-8B Q5_K_M (5.9GB) ⭐ RECOMMENDED": {
        "repo": "unsloth/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q5_K_M.gguf",
        "size": "5.9 GB",
        "type": "chat",
        "desc": "Latest Qwen3, supports /think mode"
    },
    "🔥 Qwen3-8B Q4_K_M (5.0GB) - Faster": {
        "repo": "unsloth/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q4_K_M.gguf",
        "size": "5.0 GB",
        "type": "chat",
        "desc": "Qwen3 with faster inference"
    },
    "🔥 Qwen3-4B Q8_0 (4.7GB) - Best Small": {
        "repo": "unsloth/Qwen3-4B-GGUF",
        "file": "Qwen3-4B-Q8_0.gguf",
        "size": "4.7 GB",
        "type": "chat",
        "desc": "Compact but powerful"
    },
    "🧠 DeepSeek-R1-Qwen3-8B Q5 (5.9GB) ⭐ REASONING": {
        "repo": "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        "file": "DeepSeek-R1-0528-Qwen3-8B-Q5_K_M.gguf",
        "size": "5.9 GB",
        "type": "reasoning",
        "desc": "SOTA reasoning, AIME champion"
    },
    "🧠 DeepSeek-R1-Qwen3-8B Q4 (5.0GB) - Faster": {
        "repo": "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        "file": "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
        "size": "5.0 GB",
        "type": "reasoning",
        "desc": "Fast reasoning model"
    },
    "💻 Qwen2.5-Coder-7B Q5 (5.4GB) ⭐ CODE": {
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
        "size": "5.4 GB",
        "type": "code",
        "desc": "Best open-source code model"
    },
    "💻 Qwen2.5-Coder-7B Q8 (8.1GB) - HQ Code": {
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "qwen2.5-coder-7b-instruct-q8_0.gguf",
        "size": "8.1 GB",
        "type": "code",
        "desc": "Higher quality code generation"
    },
    "💻 Qwen2.5-Coder-3B Q8 (3.6GB) - Fast Code": {
        "repo": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "file": "qwen2.5-coder-3b-instruct-q8_0.gguf",
        "size": "3.6 GB",
        "type": "code",
        "desc": "Fast coding assistant"
    },
    "🦙 Llama-3.2-3B Q5 (2.3GB)": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "size": "2.3 GB",
        "type": "chat",
        "desc": "Meta's efficient model"
    },
    "🦙 Llama-3.2-3B Q8 (3.4GB) - Higher Quality": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "size": "3.4 GB",
        "type": "chat",
        "desc": "Better quality Llama"
    },
    "🦙 Llama-3.2-1B Q8 (1.3GB) - Ultra Fast": {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "size": "1.3 GB",
        "type": "chat",
        "desc": "Fastest Llama model"
    },
    "🌟 Mistral-7B-v0.3 Q5 (5.1GB)": {
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "file": "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
        "size": "5.1 GB",
        "type": "chat",
        "desc": "Reliable general assistant"
    },
    "🌟 Qwen2.5-7B Q5 (5.4GB)": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "file": "qwen2.5-7b-instruct-q5_k_m.gguf",
        "size": "5.4 GB",
        "type": "chat",
        "desc": "Strong multilingual model"
    },
    "🌟 Phi-3.5-mini Q5 (2.8GB) - Microsoft": {
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "file": "Phi-3.5-mini-instruct-Q5_K_M.gguf",
        "size": "2.8 GB",
        "type": "chat",
        "desc": "Microsoft's efficient model"
    },
    "🌟 Gemma-2-9B Q4 (5.8GB) - Google": {
        "repo": "bartowski/gemma-2-9b-it-GGUF",
        "file": "gemma-2-9b-it-Q4_K_M.gguf",
        "size": "5.8 GB",
        "type": "chat",
        "desc": "Google's powerful model"
    },
    "⚡ SmolLM2-1.7B Q8 (1.8GB) - Tiny": {
        "repo": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "file": "SmolLM2-1.7B-Instruct-Q8_0.gguf",
        "size": "1.8 GB",
        "type": "chat",
        "desc": "Surprisingly capable tiny model"
    },
    "⚡ TinyLlama-1.1B Q8 (1.2GB) - Fastest": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "size": "1.2 GB",
        "type": "chat",
        "desc": "Minimum viable LLM"
    },
}

def get_model_list():
    return list(MODEL_CATALOG.keys())

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def download_model_hf(repo_id: str, filename: str, progress_callback=None):
    global DOWNLOAD_PROGRESS
    
    if not HF_HUB_AVAILABLE:
        return None, "huggingface_hub not installed. Run: pip install huggingface_hub"
    
    try:
        DOWNLOAD_PROGRESS["status"] = f"Downloading {filename}..."
        DOWNLOAD_PROGRESS["progress"] = 10
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        
        DOWNLOAD_PROGRESS["status"] = "Complete!"
        DOWNLOAD_PROGRESS["progress"] = 100
        
        return local_path, None
    except Exception as e:
        DOWNLOAD_PROGRESS["status"] = f"Error: {str(e)}"
        return None, str(e)

def download_model_direct(url: str, filename: str, progress_callback=None):
    global DOWNLOAD_PROGRESS
    
    output_path = MODELS_DIR / filename
    
    try:
        DOWNLOAD_PROGRESS["status"] = f"Connecting..."
        DOWNLOAD_PROGRESS["progress"] = 5
        
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get('content-length', 0))
        
        downloaded = 0
        block_size = 1024 * 1024
        
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(output_path, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        DOWNLOAD_PROGRESS["status"] = f"Downloading: {size_mb:.0f}/{total_mb:.0f} MB"
                        DOWNLOAD_PROGRESS["progress"] = progress
        
        DOWNLOAD_PROGRESS["status"] = "Complete!"
        DOWNLOAD_PROGRESS["progress"] = 100
        return str(output_path), None
        
    except Exception as e:
        DOWNLOAD_PROGRESS["status"] = f"Error: {str(e)}"
        if output_path.exists():
            output_path.unlink()
        return None, str(e)

def ui_download_model(model_selection, custom_repo, custom_file):
    global DOWNLOAD_PROGRESS
    DOWNLOAD_PROGRESS = {"status": "Starting...", "progress": 0}
    
    if custom_repo.strip() and custom_file.strip():
        repo_id = custom_repo.strip()
        filename = custom_file.strip()
        model_name = filename
    elif model_selection and model_selection in MODEL_CATALOG:
        model_info = MODEL_CATALOG[model_selection]
        if model_info is None:
            return "❌ Please select a model", ""
        repo_id = model_info["repo"]
        filename = model_info["file"]
        model_name = model_selection
    else:
        return "❌ Please select a model or enter custom repo/file", ""
    
    local_path = MODELS_DIR / filename
    if local_path.exists():
        return f"✅ Model already exists!\n📁 {local_path}", str(local_path)
    
    yield f"⏳ Downloading: {model_name}\n📦 {repo_id}\n📄 {filename}\n\nThis may take several minutes...", ""
    
    if HF_HUB_AVAILABLE:
        path, error = download_model_hf(repo_id, filename)
    else:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        path, error = download_model_direct(url, filename)
    
    if error:
        yield f"❌ Download failed: {error}", ""
    else:
        final_path = MODELS_DIR / filename
        yield f"✅ Download complete!\n📁 {final_path}\n\n💡 Click 'Use This Model' to load it", str(final_path)

def ui_use_downloaded(model_path):
    if model_path:
        return model_path
    return ""

def ui_list_local_models():
    if not MODELS_DIR.exists():
        return "No models directory found"
    
    models = list(MODELS_DIR.glob("*.gguf"))
    if not models:
        return "📭 No GGUF models downloaded yet\n\nSelect a model above and click Download!"
    
    result = "📁 Downloaded Models:\n" + "─" * 35 + "\n"
    for m in sorted(models):
        size_gb = m.stat().st_size / (1024**3)
        result += f"• {m.name}\n  ({size_gb:.1f} GB)\n"
    
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════
CHAT_TEMPLATES = {
    "None (Raw)": {
        "format": "{prompt}",
        "description": "No template - raw prompt",
        "detect": []
    },
    "Qwen3": {
        "format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "description": "Qwen3 (supports /think)",
        "detect": ["qwen3"]
    },
    "Llama-2/Mistral": {
        "format": "[INST] {prompt} [/INST]",
        "description": "Llama-2, Mistral",
        "detect": ["llama-2", "mistral", "mixtral", "inst"]
    },
    "Llama-3": {
        "format": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "description": "Llama-3-Instruct",
        "detect": ["llama-3", "llama3"]
    },
    "ChatML (Qwen2/Yi)": {
        "format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "description": "Qwen2, Yi, OpenHermes",
        "detect": ["qwen2", "qwen-", "yi-", "chatml", "hermes"]
    },
    "DeepSeek": {
        "format": "### Instruction:\n{prompt}\n\n### Response:\n",
        "description": "DeepSeek-Coder",
        "detect": ["deepseek-coder"]
    },
    "DeepSeek-V2/V3/R1": {
        "format": "<|begin▁of▁sentence|><|User|>{prompt}<|Assistant|>",
        "description": "DeepSeek-V2/V3/R1",
        "detect": ["deepseek-v2", "deepseek-v3", "deepseek-r1"]
    },
    "Phi-3": {
        "format": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        "description": "Microsoft Phi-3",
        "detect": ["phi-3", "phi3"]
    },
    "Gemma": {
        "format": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "description": "Google Gemma",
        "detect": ["gemma"]
    },
    "Alpaca": {
        "format": "### Instruction:\n{prompt}\n\n### Response:\n",
        "description": "Alpaca-style",
        "detect": ["alpaca"]
    },
    "Vicuna": {
        "format": "USER: {prompt}\nASSISTANT:",
        "description": "Vicuna",
        "detect": ["vicuna"]
    },
}

def detect_template(model_path: str) -> str:
    filename = os.path.basename(model_path).lower()
    
    priority_order = [
        "Qwen3",
        "DeepSeek-V2/V3/R1",
        "DeepSeek",
        "Llama-3",
        "Phi-3",
        "ChatML (Qwen2/Yi)",
        "Gemma",
        "Vicuna",
        "Alpaca",
        "Llama-2/Mistral",
    ]
    
    for template_name in priority_order:
        if template_name not in CHAT_TEMPLATES:
            continue
        for pattern in CHAT_TEMPLATES[template_name]["detect"]:
            if pattern in filename:
                return template_name
    
    return "None (Raw)"

def apply_template(prompt: str, template_name: str) -> str:
    if template_name not in CHAT_TEMPLATES:
        return prompt
    template = CHAT_TEMPLATES[template_name]["format"]
    return template.format(prompt=prompt)

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def get_gpu_info():
    if not torch.cuda.is_available():
        return [{"id": -1, "name": "CPU Only", "memory": 0, "compute": "N/A"}]
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "id": i,
            "name": props.name,
            "memory": props.total_memory / 1e9,
            "compute": f"{props.major}.{props.minor}",
            "is_modern": props.major >= 8
        })
    return gpus

def get_gpu_stats(device_id=0):
    if not torch.cuda.is_available():
        return "CPU Mode"
    
    try:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9
        total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        
        usage_pct = (allocated / total) * 100
        bar_len = 15
        filled = int(bar_len * usage_pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        return f"[{bar}] {allocated:.1f}/{total:.1f}GB ({usage_pct:.0f}%)"
    except Exception as e:
        return f"Error: {e}"

def get_device_config(device_id):
    if not torch.cuda.is_available():
        return "cpu", "float32", contextlib.nullcontext()
    
    device = f"cuda:{device_id}"
    try:
        props = torch.cuda.get_device_properties(device)
        
        if props.major < 8:
            dtype = "float16"
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                attn = sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
            except ImportError:
                import warnings
                warnings.filterwarnings("ignore", message=".*sdp_kernel.*deprecated.*")
                attn = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, 
                    enable_math=True, 
                    enable_mem_efficient=True
                )
        else:
            dtype = "bfloat16"
            attn = contextlib.nullcontext()
            
        return device, dtype, attn
    except Exception as e:
        print(f"❌ GPU {device_id} Error: {e}")
        return "cpu", "float32", contextlib.nullcontext()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_native(path, device_idx):
    global CURRENT_MODEL, ENC, CURRENT_ENGINE, MODEL_INFO
    
    if not NATIVE_AVAILABLE:
        return "❌ Native engine unavailable (model_llama3.py not found)"
    
    device, dtype, _ = get_device_config(device_idx)
    
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        if isinstance(ckpt['model_config'], dict):
            conf = GPTConfig(**ckpt['model_config'])
        else:
            conf = ckpt['model_config']
        
        model = GPT(conf)
        
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state_dict)
        
        if dtype == 'float16':
            model.half()
        elif dtype == 'bfloat16':
            model.bfloat16()
        
        model.to(device)
        model.eval()
        
        params = sum(p.numel() for p in model.parameters())
        
        CURRENT_MODEL = model
        ENC = tiktoken.get_encoding("gpt2")
        CURRENT_ENGINE = "native"
        MODEL_INFO = {
            "name": os.path.basename(path),
            "params": f"{params/1e6:.1f}M",
            "layers": conf.n_layer,
            "heads": conf.n_head,
            "ctx": conf.block_size,
            "dtype": dtype,
            "device": device
        }
        
        return f"""✅ Model Loaded Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 {MODEL_INFO['name']}
🔢 {MODEL_INFO['params']} parameters
📊 {MODEL_INFO['layers']} layers, {MODEL_INFO['heads']} heads
📐 {MODEL_INFO['ctx']} context
⚡ {dtype.upper()} on {device}"""
        
    except Exception as e:
        return f"❌ Load Failed: {str(e)}"

def load_gguf(path, device_idx, n_ctx):
    global CURRENT_MODEL, CURRENT_ENGINE, MODEL_INFO
    
    if not GGUF_AVAILABLE:
        return "❌ GGUF unavailable (pip install llama-cpp-python)"
    
    try:
        model = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            main_gpu=device_idx,
            verbose=False
        )
        
        CURRENT_MODEL = model
        CURRENT_ENGINE = "gguf"
        MODEL_INFO = {
            "name": os.path.basename(path),
            "params": "Unknown",
            "ctx": n_ctx,
            "device": f"cuda:{device_idx}"
        }
        
        return f"""✅ GGUF Model Loaded
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 {MODEL_INFO['name']}
📐 {n_ctx} context length
🖥️ GPU {device_idx}"""
        
    except Exception as e:
        return f"❌ GGUF Load Failed: {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def stop_generation():
    global STOP_GENERATION
    STOP_GENERATION = True
    return "⏹️ Stopping..."

def generate(prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, template_name, device_idx):
    global STOP_GENERATION
    STOP_GENERATION = False
    
    if not CURRENT_MODEL:
        yield "⚠️ No model loaded. Please load a model first."
        return
    
    if not prompt.strip():
        yield "⚠️ Please enter a prompt"
        return
    
    formatted_prompt = apply_template(prompt, template_name)
    
    start_time = time.time()
    token_count = 0
    
    if CURRENT_ENGINE == "gguf":
        try:
            stream = CURRENT_MODEL(
                formatted_prompt,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repeat_penalty=float(repeat_penalty),
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=True,
                stop=["</s>", "<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "\n\n\n"]
            )
            
            partial = ""
            for output in stream:
                if STOP_GENERATION:
                    partial += "\n\n⏹️ [Stopped]"
                    yield partial
                    return
                    
                token = output['choices'][0]['text']
                partial += token
                token_count += 1
                yield partial
                
        except Exception as e:
            yield f"❌ Generation Error: {str(e)}"
            return
    else:
        device, _, attn_ctx = get_device_config(int(device_idx))
        
        try:
            tokens = ENC.encode(formatted_prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            
            partial = formatted_prompt
            
            CURRENT_MODEL.eval()
            with torch.no_grad(), attn_ctx:
                for _ in range(int(max_tokens)):
                    if STOP_GENERATION:
                        partial += "\n\n⏹️ [Stopped]"
                        yield partial
                        return
                    
                    block_size = CURRENT_MODEL.config.block_size
                    idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
                    
                    logits, _ = CURRENT_MODEL(idx_cond)
                    logits = logits[:, -1, :] / float(temperature)
                    
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float('-inf')
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    token_str = ENC.decode([idx_next.item()])
                    partial += token_str
                    token_count += 1
                    
                    idx = torch.cat((idx, idx_next), dim=1)
                    yield partial
                    
        except Exception as e:
            yield f"❌ Generation Error: {str(e)}"
            return
    
    elapsed = time.time() - start_time
    tps = token_count / elapsed if elapsed > 0 else 0
    yield partial + f"\n\n─────────────────────\n⚡ {token_count} tokens • {elapsed:.1f}s • {tps:.1f} tok/s"

# ═══════════════════════════════════════════════════════════════════════════════
# UI HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════
def ui_load(engine, path, gpu, ctx):
    global CURRENT_TEMPLATE
    
    if not path.strip():
        return "❌ No path specified", "None (Raw)"
    if not os.path.exists(path):
        return f"❌ File not found: {path}", "None (Raw)"
    
    detected = detect_template(path)
    CURRENT_TEMPLATE = detected
    template_msg = f"\n💬 Template: {detected}" if detected != "None (Raw)" else ""
    
    path_lower = path.lower()
    if path_lower.endswith('.gguf'):
        if engine == "Native (.pt)":
            msg = load_gguf(path, int(gpu), int(ctx)) + "\n⚠️ Auto-switched to GGUF" + template_msg
            return msg, detected
        return load_gguf(path, int(gpu), int(ctx)) + template_msg, detected
    elif path_lower.endswith('.pt') or path_lower.endswith('.pth'):
        if engine == "GGUF (.gguf)":
            msg = load_native(path, int(gpu)) + "\n⚠️ Auto-switched to Native" + template_msg
            return msg, detected
        return load_native(path, int(gpu)) + template_msg, detected
    else:
        if engine == "Native (.pt)":
            return load_native(path, int(gpu)) + template_msg, detected
        else:
            return load_gguf(path, int(gpu), int(ctx)) + template_msg, detected

def ui_get_stats(gpu):
    return get_gpu_stats(int(gpu))

def get_system_info():
    gpus = get_gpu_info()
    info = "🖥️ System Info\n" + "─" * 30 + "\n"
    
    for gpu in gpus:
        if gpu['id'] == -1:
            info += "CPU Mode Active\n"
        else:
            status = "⚡" if gpu.get('is_modern', False) else "⚠️"
            info += f"GPU {gpu['id']}: {gpu['name'][:20]}\n"
            info += f"  {gpu['memory']:.1f}GB • SM {gpu['compute']} {status}\n"
    
    return info

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE FUNCTIONS (Placeholder - requires browser API)
# ═══════════════════════════════════════════════════════════════════════════════
def process_voice_input(audio):
    """Process voice input - placeholder for speech-to-text"""
    if audio is None:
        return ""
    # In a full implementation, this would use Whisper or similar
    return "[Voice input received - Speech-to-text processing would happen here]"

def generate_voice_output(text):
    """Generate voice output - placeholder for TTS"""
    # In a full implementation, this would use TTS
    return f"🔊 Would speak: {text[:100]}..."

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS (Placeholder)
# ═══════════════════════════════════════════════════════════════════════════════
def start_training(dataset_path, epochs, batch_size, learning_rate, save_path):
    """Start training - placeholder"""
    global TRAINING_ACTIVE
    
    if not dataset_path.strip():
        return "❌ Please specify a dataset path"
    
    TRAINING_ACTIVE = True
    
    result = f"""🏋️ Training Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 Dataset: {dataset_path}
🔄 Epochs: {epochs}
📦 Batch Size: {batch_size}
📈 Learning Rate: {learning_rate}
💾 Save Path: {save_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Training interface is a placeholder.
To actually train, use train_llama3.py directly:

python train_llama3.py --epochs {epochs} --batch_size {batch_size}

Full training integration coming in v3.1!"""
    
    TRAINING_ACTIVE = False
    return result

def stop_training():
    """Stop training"""
    global TRAINING_ACTIVE
    TRAINING_ACTIVE = False
    return "⏹️ Training stopped"

# ═══════════════════════════════════════════════════════════════════════════════
# CSS - Modern, Responsive, Theme-aware
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
/* Fonts loaded via HTML link tag instead of @import */

:root {
    --bg-primary: #0f1419;
    --bg-secondary: #1a1f2e;
    --bg-tertiary: #242b3d;
    --bg-card: #1e2433;
    --bg-input: #161b26;
    --text-primary: #e8eaed;
    --text-secondary: #9aa0a6;
    --text-muted: #5f6368;
    --accent-primary: #00d4ff;
    --accent-secondary: #7c4dff;
    --accent-success: #00e676;
    --accent-warning: #ffab00;
    --accent-danger: #ff5252;
    --border-color: rgba(255, 255, 255, 0.1);
    --border-glow: rgba(0, 212, 255, 0.3);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}

.light-theme {
    --bg-primary: #f8f9fa;
    --bg-secondary: #ffffff;
    --bg-tertiary: #e9ecef;
    --bg-card: #ffffff;
    --bg-input: #f1f3f4;
    --text-primary: #202124;
    --text-secondary: #5f6368;
    --text-muted: #9aa0a6;
    --accent-primary: #0066cc;
    --accent-secondary: #6200ee;
    --border-color: rgba(0, 0, 0, 0.1);
    --border-glow: rgba(0, 102, 204, 0.3);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.12);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.16);
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
    padding: 16px !important;
}

.header-container {
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: var(--shadow-md);
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    margin: 0;
}

.header-badges {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border-radius: 20px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}

.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 20px !important;
    box-shadow: var(--shadow-sm);
}

.section-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent-primary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
}

.gradio-textbox textarea,
.gradio-textbox input,
input[type="text"],
textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.95rem !important;
    padding: 12px !important;
}

.gradio-textbox textarea:focus,
.gradio-textbox input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.15) !important;
}

label, .label-wrap span {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

.gradio-dropdown select,
.gradio-dropdown input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    padding: 12px !important;
    font-size: 0.95rem !important;
}

.gradio-dropdown ul,
.gradio-dropdown .options {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-lg) !important;
    max-height: 300px !important;
    overflow-y: auto !important;
    z-index: 9999 !important;
}

.gradio-dropdown li,
.gradio-dropdown .option {
    padding: 12px 16px !important;
    color: var(--text-primary) !important;
    cursor: pointer;
    border-bottom: 1px solid var(--border-color);
}

.gradio-dropdown li:hover,
.gradio-dropdown .option:hover {
    background: var(--bg-tertiary) !important;
}

.gradio-radio {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}

.gradio-radio label {
    display: flex !important;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    cursor: pointer;
}

.gradio-radio label:hover {
    border-color: var(--accent-primary);
}

.gradio-radio input[type="radio"] {
    accent-color: var(--accent-primary);
}

.gradio-radio label span {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

.gradio-slider input[type="range"] {
    accent-color: var(--accent-primary);
}

button, .btn {
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}

.btn-primary, button.primary {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
    border: none !important;
    color: white !important;
}

.btn-primary:hover, button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
}

.btn-secondary {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

.btn-danger {
    background: rgba(255, 82, 82, 0.15) !important;
    border: 1px solid var(--accent-danger) !important;
    color: var(--accent-danger) !important;
}

.btn-success {
    background: rgba(0, 230, 118, 0.15) !important;
    border: 1px solid var(--accent-success) !important;
    color: var(--accent-success) !important;
}

.output-box textarea {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    color: var(--accent-success) !important;
    font-family: var(--font-mono) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    padding: 20px !important;
}

.status-box textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--accent-primary) !important;
    font-family: var(--font-mono) !important;
}

.gradio-accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 16px;
}

.gradio-accordion summary {
    background: var(--bg-tertiary) !important;
    padding: 14px 20px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.gradio-tabs > div:first-child {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    padding: 8px !important;
    border: 1px solid var(--border-color) !important;
    border-bottom: none !important;
    gap: 8px !important;
    flex-wrap: wrap;
}

.gradio-tabs button {
    background: transparent !important;
    border: none !important;
    color: var(--text-secondary) !important;
    padding: 12px 20px !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
}

.gradio-tabs button.selected {
    background: var(--bg-card) !important;
    color: var(--accent-primary) !important;
}

.gradio-tabs > div:last-child {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    padding: 24px !important;
}

@media (max-width: 768px) {
    .gradio-container { padding: 12px !important; }
    .header-title { font-size: 1.8rem; }
    .gradio-row { flex-direction: column !important; }
    .gradio-column { width: 100% !important; max-width: 100% !important; }
    .card { padding: 16px !important; }
    .gradio-tabs button { padding: 10px 16px !important; font-size: 0.9rem !important; }
    button, .btn { padding: 10px 16px !important; }
}

@media (max-width: 480px) {
    .header-title { font-size: 1.5rem; }
    .gradio-radio { flex-direction: column; }
    .gradio-radio label { width: 100%; }
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }
"""

THEME_JS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
    #theme-toggle-btn {
        position: fixed;
        top: 16px;
        right: 16px;
        z-index: 1000;
        background: var(--bg-card, #1e2433);
        border: 1px solid var(--border-color, rgba(255,255,255,0.1));
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        font-size: 1.2rem;
        transition: all 0.3s ease;
        user-select: none;
    }
    #theme-toggle-btn:hover {
        transform: scale(1.1);
        border-color: var(--accent-primary, #00d4ff);
    }
    @media (max-width: 768px) {
        #theme-toggle-btn {
            top: auto;
            bottom: 16px;
            width: 44px;
            height: 44px;
        }
    }
</style>
"""

# Theme toggle button with inline JavaScript (Gradio doesn't execute <script> tags)
THEME_TOGGLE_HTML = '''
<div id="theme-toggle-btn" onclick="
    var body = document.body;
    var container = document.querySelector('.gradio-container');
    var btn = this;
    if (body.classList.contains('light-theme')) {
        body.classList.remove('light-theme');
        if (container) container.classList.remove('light-theme');
        btn.textContent = '🌙';
        localStorage.setItem('sovra-theme', 'dark');
    } else {
        body.classList.add('light-theme');
        if (container) container.classList.add('light-theme');
        btn.textContent = '☀️';
        localStorage.setItem('sovra-theme', 'light');
    }
">🌙</div>
'''

# JavaScript to run on page load (Gradio executes this properly)
LOAD_THEME_JS = """
function() {
    // Load saved theme
    var savedTheme = localStorage.getItem('sovra-theme');
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        var container = document.querySelector('.gradio-container');
        if (container) container.classList.add('light-theme');
        var btn = document.getElementById('theme-toggle-btn');
        if (btn) btn.textContent = '☀️';
    }
    return [];
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(css=CSS, title="SOVRA OMNI v3.0", theme=gr.themes.Base(), js=LOAD_THEME_JS) as demo:
    
    gr.HTML(THEME_JS + THEME_TOGGLE_HTML)
    
    gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">SOVRA OMNI</h1>
            <p class="header-subtitle">Neural Interface for LLM Inference • v3.0</p>
            <div class="header-badges">
                <span class="badge">🚀 GGUF Support</span>
                <span class="badge">🎯 Native .pt Models</span>
                <span class="badge">📥 Model Downloader</span>
                <span class="badge">🏋️ Training</span>
            </div>
        </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("💬 Chat", id="chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">⚡ Model Setup</div>')
                    
                    with gr.Group(elem_classes="card"):
                        engine_radio = gr.Radio(
                            ["Native (.pt)", "GGUF (.gguf)"],
                            label="Engine Type",
                            value="GGUF (.gguf)"
                        )
                        
                        model_path = gr.Textbox(
                            label="Model Path",
                            value="models/",
                            placeholder="/path/to/model.gguf"
                        )
                        
                        with gr.Row():
                            gpu_dropdown = gr.Dropdown(
                                choices=[str(i) for i in range(max(1, torch.cuda.device_count()))],
                                value=str(ARGS_DEVICE),
                                label="GPU"
                            )
                            ctx_slider = gr.Slider(
                                512, 8192, value=4096, step=256,
                                label="Context Length"
                            )
                        
                        load_btn = gr.Button("⚡ Load Model", variant="primary", elem_classes="btn-primary")
                    
                    gr.HTML('<div class="section-title">📊 Status</div>')
                    with gr.Group(elem_classes="card"):
                        status_box = gr.Textbox(label="", lines=6, interactive=False, elem_classes="status-box", value=get_system_info())
                        gpu_stats = gr.Textbox(label="VRAM Usage", interactive=False, value=get_gpu_stats(ARGS_DEVICE))
                        refresh_btn = gr.Button("🔄 Refresh", size="sm", elem_classes="btn-secondary")
                
                with gr.Column(scale=2):
                    gr.HTML('<div class="section-title">🧠 Output</div>')
                    
                    with gr.Group(elem_classes="card"):
                        output_box = gr.Textbox(label="", lines=16, interactive=False, elem_classes="output-box", placeholder="Model response will appear here...")
                    
                    gr.HTML('<div class="section-title">📝 Input</div>')
                    with gr.Group(elem_classes="card"):
                        input_box = gr.Textbox(label="", lines=3, placeholder="Type your message here... (Enter to send)")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                temp_slider = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")
                            with gr.Column(scale=1):
                                max_tokens_slider = gr.Slider(10, 4096, value=512, step=10, label="Max Tokens")
                        
                        with gr.Row():
                            top_k_slider = gr.Slider(0, 500, value=40, step=5, label="Top-K")
                            top_p_slider = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-P")
                            repeat_penalty_slider = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repeat Penalty")
                        
                        template_dropdown = gr.Dropdown(choices=list(CHAT_TEMPLATES.keys()), value="None (Raw)", label="💬 Chat Template")
                        
                        with gr.Row():
                            generate_btn = gr.Button("🚀 Generate", variant="primary", elem_classes="btn-primary")
                            stop_btn = gr.Button("⏹️ Stop", elem_classes="btn-danger")
                            clear_btn = gr.Button("🗑️ Clear", elem_classes="btn-secondary")
        
        with gr.Tab("📥 Download Models", id="download"):
            gr.HTML('<div class="section-title">Download GGUF Models from HuggingFace</div>')
            
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group(elem_classes="card"):
                        model_select = gr.Dropdown(choices=get_model_list(), label="🎯 Select Model", info="Scroll to see all options")
                        
                        gr.HTML('<p style="color: var(--text-secondary); margin: 16px 0 8px;">Or enter custom repo:</p>')
                        
                        with gr.Row():
                            custom_repo = gr.Textbox(label="Repo ID", placeholder="username/repo-name", scale=2)
                            custom_file = gr.Textbox(label="Filename", placeholder="model-Q5_K_M.gguf", scale=2)
                        
                        with gr.Row():
                            download_btn = gr.Button("⬇️ Download Model", variant="primary", elem_classes="btn-primary")
                            list_local_btn = gr.Button("📁 Show Downloaded", elem_classes="btn-secondary")
                
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        download_status = gr.Textbox(label="Status", lines=10, interactive=False, value=ui_list_local_models())
                        downloaded_path = gr.Textbox(visible=False)
                        use_model_btn = gr.Button("✅ Use This Model", elem_classes="btn-success")
        
        with gr.Tab("🏋️ Training", id="training"):
            gr.HTML('<div class="section-title">Model Training Interface</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="section-title">📂 Data Configuration</div>')
                        train_dataset = gr.Textbox(label="Dataset Path", placeholder="data/train.jsonl", value="data/fineweb/")
                        train_output = gr.Textbox(label="Output Path", placeholder="checkpoints/my_model", value="checkpoints/")
                        
                        with gr.Row():
                            train_epochs = gr.Slider(1, 100, value=3, step=1, label="Epochs")
                            train_batch = gr.Slider(1, 64, value=4, step=1, label="Batch Size")
                        
                        train_lr = gr.Slider(1e-6, 1e-3, value=3e-4, step=1e-6, label="Learning Rate")
                        
                        with gr.Row():
                            train_start_btn = gr.Button("🚀 Start Training", variant="primary", elem_classes="btn-primary")
                            train_stop_btn = gr.Button("⏹️ Stop", elem_classes="btn-danger")
                
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="section-title">📊 Training Status</div>')
                        train_status = gr.Textbox(label="", lines=15, interactive=False, value="Ready to train.\n\nConfigure settings and click 'Start Training'.")
        
        with gr.Tab("⚙️ Settings", id="settings"):
            gr.HTML('<div class="section-title">Application Settings</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="section-title">🎨 Appearance</div>')
                        gr.HTML('<p style="color: var(--text-secondary);">Click the 🌙/☀️ button in the corner to toggle theme.</p>')
                        
                        gr.HTML('<div class="section-title">🔊 Voice Settings</div>')
                        gr.HTML('''<p style="color: var(--text-secondary); margin-bottom: 12px;">
                            ⚠️ <strong>Voice features require HTTPS</strong><br>
                            Access via <code>https://</code> or <code>localhost</code> to enable microphone.<br>
                            Current access via HTTP will block microphone access.
                        </p>''')
                        voice_enabled = gr.Checkbox(label="Enable Voice Features (requires HTTPS)", value=False)
                        voice_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="TTS Speed")
                
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="section-title">🖥️ System Info</div>')
                        system_info = gr.Textbox(label="", lines=10, interactive=False, value=get_system_info())
                        gr.HTML(f"""<div style="color: var(--text-secondary); margin-top: 16px;">
                            <p><strong>Version:</strong> SOVRA OMNI v3.0</p>
                            <p><strong>PyTorch:</strong> {torch.__version__}</p>
                            <p><strong>CUDA:</strong> {torch.cuda.is_available()}</p>
                        </div>""")
    
    # Event Handlers
    load_btn.click(ui_load, inputs=[engine_radio, model_path, gpu_dropdown, ctx_slider], outputs=[status_box, template_dropdown])
    refresh_btn.click(ui_get_stats, inputs=[gpu_dropdown], outputs=gpu_stats)
    generate_btn.click(generate, inputs=[input_box, max_tokens_slider, temp_slider, top_k_slider, top_p_slider, repeat_penalty_slider, template_dropdown, gpu_dropdown], outputs=output_box)
    input_box.submit(generate, inputs=[input_box, max_tokens_slider, temp_slider, top_k_slider, top_p_slider, repeat_penalty_slider, template_dropdown, gpu_dropdown], outputs=output_box)
    stop_btn.click(stop_generation, outputs=status_box)
    clear_btn.click(lambda: ("", ""), outputs=[input_box, output_box])
    download_btn.click(ui_download_model, inputs=[model_select, custom_repo, custom_file], outputs=[download_status, downloaded_path])
    list_local_btn.click(ui_list_local_models, outputs=download_status)
    use_model_btn.click(ui_use_downloaded, inputs=[downloaded_path], outputs=model_path)
    train_start_btn.click(start_training, inputs=[train_dataset, train_epochs, train_batch, train_lr, train_output], outputs=train_status)
    train_stop_btn.click(stop_training, outputs=train_status)

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SOVRA OMNI v3.0 - Starting...                                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Port:   {args.port:<6}                                                       ║
║  Share:  {str(args.share):<6}                                                 ║
║  GPU:    {ARGS_DEVICE:<6}                                                     ║
║  Models: {str(MODELS_DIR):<50}  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
