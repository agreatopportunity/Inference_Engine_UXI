"""
Universal LLM Playground (2025 Edition)
Compatible with: Titan V, RTX 3090, 4090, 5090
"""
import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
import contextlib
import argparse
import os

# Import your architecture
from model_llama3 import GPT, GPTConfig

# --- HARDWARE AUTO-DETECTION ---
def get_optimal_device_settings(device_id=0):
    """
    Automatically determines the best precision and attention mechanisms
    based on the specific GPU architecture detected.
    """
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected. Using CPU (this will be slow).")
        return "cpu", "float32", contextlib.nullcontext()

    device = f"cuda:{device_id}"
    props = torch.cuda.get_device_properties(device)
    name = props.name
    major_version = props.major
    
    print(f"\n[Hardware Detected] {name} (Compute {major_version}.{props.minor})")
    
    # 1. Determine Precision (DType)
    # Ampere (8.0) and newer support bfloat16 (RTX 30xx, 40xx, 50xx, A100, H100)
    if major_version >= 8:
        dtype = "bfloat16"
        print("   ✅ Architecture supports bfloat16. Using modern precision.")
    else:
        # Volta (7.0) or Pascal (6.1) do not support bfloat16
        dtype = "float16"
        print("   ⚠️  Legacy architecture detected (Titan V / Pascal). Switching to float16 to prevent crashes.")

    # 2. Determine Attention Backend
    # Flash Attention requires Compute Capability 8.0+
    if major_version >= 8:
        # Default PyTorch behavior prefers Flash Attention on supported hardware
        attn_context = contextlib.nullcontext()
        print("   ✅ Flash Attention enabled.")
    else:
        # Force disable Flash Attention for older cards
        attn_context = torch.backends.cuda.sdp_kernel(
            enable_flash=False, 
            enable_math=True, 
            enable_mem_efficient=True
        )
        print("   ⚠️  Flash Attention disabled (Legacy compatibility mode).")
        
    return device, dtype, attn_context

# --- SETUP GLOBALS ---
# We use a placeholder here; these get populated in the main block
DEVICE = "cuda"
DTYPE = "float16"
ATTN_CONTEXT = contextlib.nullcontext()
CHECKPOINT = "checkpoints/latest.pt"

# --- MODEL LOADER ---
def load_model(device, dtype):
    print(f"\nLoading model from {CHECKPOINT}...")
    
    # 1. Load to CPU first (Critical for cross-GPU compatibility)
    try:
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        return None, f"Checkpoint not found at {CHECKPOINT}"

    # 2. Load Config
    if isinstance(checkpoint['model_config'], dict):
        config = GPTConfig(**checkpoint['model_config'])
    else:
        config = checkpoint['model_config']
    
    print(f"   Config: {config.n_layer} layers, {config.n_head} heads")
    model = GPT(config)
    
    # 3. Clean State Dict (Remove torch.compile '_orig_mod.' prefix)
    state_dict = checkpoint['model']
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    # 4. Cast Precision
    if dtype == "bfloat16":
        model.bfloat16()
    elif dtype == "float16":
        model.half()
    else:
        model.float()
        
    # 5. Move to GPU
    print(f"   Moving model to {device}...")
    model.to(device)
    model.eval()
    return model, None

# --- GENERATION FUNCTION ---
def generate_text(prompt, max_tokens, temperature, top_k):
    if model is None:
        return "Error: Model failed to load. Check console."

    # Encode
    encoded = enc.encode(prompt)
    idx = torch.tensor(encoded).unsqueeze(0).to(DEVICE)
    
    generated_text = prompt
    
    # Run generation
    model.eval()
    # Use the hardware-specific attention context we detected earlier
    with torch.no_grad(), ATTN_CONTEXT:
        for _ in range(max_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Decode token
            new_token = enc.decode([idx_next.item()])
            generated_text += new_token
            
            # Update sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            yield generated_text

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0, help="Which GPU index to use (default 0)")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/latest.pt")
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    
    # 1. Run Auto-Detection
    DEVICE, DTYPE, ATTN_CONTEXT = get_optimal_device_settings(args.device_id)
    
    # 2. Load Resources
    model, error_msg = load_model(DEVICE, DTYPE)
    
    if model:
        enc = tiktoken.get_encoding("gpt2")
        print("✅ System Ready!")
        
        # 3. Launch UI
        gpu_name = torch.cuda.get_device_name(DEVICE)
        
        with gr.Blocks(theme=gr.themes.Soft(), title=f"LLM Lab - {gpu_name}") as demo:
            gr.Markdown(f"# 🧠 Universal LLM Playground")
            gr.Markdown(f"**Hardware:** {gpu_name} | **Precision:** {DTYPE} | **Attention:** {'Standard' if DTYPE == 'bfloat16' else 'Legacy (Efficient)'}")
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(label="Prompt", placeholder="The future of AI is...", lines=5)
                    with gr.Accordion("Settings", open=True):
                        max_tokens = gr.Slider(10, 1000, value=200, step=10, label="Max Tokens")
                        temperature = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                        top_k = gr.Slider(0, 500, value=200, label="Top-K")
                    generate_btn = gr.Button("🚀 Generate", variant="primary")
                
                with gr.Column(scale=1):
                    output_text = gr.Textbox(label="Output", lines=20, interactive=False)

            generate_btn.click(
                fn=generate_text,
                inputs=[prompt_input, max_tokens, temperature, top_k],
                outputs=output_text
            )
            
        demo.launch(server_name="0.0.0.0", share=True)
    else:
        print(f"❌ Failed to start: {error_msg}")
