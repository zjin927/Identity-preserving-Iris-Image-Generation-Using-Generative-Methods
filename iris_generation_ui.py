"""
Simple Gradio UI for iris image generation.
Allows selecting a reference image and adjusting the strength parameter.
"""
import os
import sys
import tempfile
import gradio as gr
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "latent-diffusion"))
# Add taming-transformers to path so taming module can be imported
sys.path.insert(0, str(project_root / "latent-diffusion" / "taming-transformers"))

# Import the generation function
import importlib.util
spec = importlib.util.spec_from_file_location(
    "img2img_single_image",
    project_root / "latent-diffusion" / "img2img_single_image.py"
)
img2img_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(img2img_module)
generate_single_image = img2img_module.generate_single_image


# Default paths (user should modify these)
DEFAULT_CONFIG = "local_configs/latent-diffusion/iris_stageA_sota_64.yaml"
DEFAULT_CKPT = "models/best_ldm_model.ckpt"  # User needs to set this
DEFAULT_CLASS_ID = 0  # Default class ID

# Model cache (global variables to cache loaded model)
_cached_model = None
_cached_sampler = None
_cached_cfg = None
_cached_config_path = None
_cached_ckpt_path = None


def load_model_if_needed(config_path, ckpt_path):
    """Load model if not cached or if paths changed."""
    global _cached_model, _cached_sampler, _cached_cfg, _cached_config_path, _cached_ckpt_path
    
    # Check if we need to reload model
    if (_cached_model is None or 
        _cached_config_path != config_path or 
        _cached_ckpt_path != ckpt_path):
        
        # Import load_model from the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "img2img_single_image",
            project_root / "latent-diffusion" / "img2img_single_image.py"
        )
        img2img_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(img2img_module)
        load_model = img2img_module.load_model
        
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cached_model, _cached_sampler, _cached_cfg = load_model(config_path, ckpt_path, device)
        _cached_config_path = config_path
        _cached_ckpt_path = ckpt_path
        return "Model loaded to cache"
    
    return "Using cached model"


def generate_image(
    ref_image,
    mask_image,
    strength,
    config_path,
    ckpt_path,
    class_id
):
    """
    Generate synthetic image from reference image.
    
    Args:
        ref_image: PIL Image or file path
        mask_image: PIL Image or file path
        strength: float, noise injection depth (0-1)
        config_path: str, path to config YAML
        ckpt_path: str, path to checkpoint
        class_id: int, class ID for conditioning
    
    Returns:
        Generated image path
    """
    global _cached_model, _cached_sampler, _cached_cfg
    
    if ref_image is None:
        return None, "Please select a reference image"
    if mask_image is None:
        return None, "Please select a mask image"
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, "Please set a valid model checkpoint path"
    if not config_path or not os.path.exists(config_path):
        return None, "Please set a valid config file path"
    
    try:
        # Load model if needed (with caching)
        load_status = load_model_if_needed(config_path, ckpt_path)
        print(f">> {load_status}")
        # Save uploaded images to temp files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_ref:
            ref_path = tmp_ref.name
            if isinstance(ref_image, str):
                import shutil
                shutil.copy(ref_image, ref_path)
            else:
                ref_image.save(ref_path)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
            mask_path = tmp_mask.name
            if isinstance(mask_image, str):
                import shutil
                shutil.copy(mask_image, mask_path)
            else:
                mask_image.save(mask_path)
        
        # Create output directory
        output_dir = os.path.join(project_root, "outputs", "ui_generated")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        import time
        output_filename = f"generated_{int(time.time())}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Fixed parameters (optimized for speed on RTX 2060)
        # Reduced denoise_steps from 50 to 30 for faster generation (can adjust quality vs speed)
        ddim_steps = 100
        denoise_steps = 50  # Reduced from 50 for faster generation
        eta = 0.0
        mask_dilate_px = 2
        mask_feather_px = 3
        seed = 0
        
        # Generate image (using cached model)
        generate_single_image(
            config_path=config_path,
            ckpt_path=ckpt_path,
            ref_img_path=ref_path,
            mask_path=mask_path,
            class_id=int(class_id),
            out_path=output_path,
            strength=float(strength),
            ddim_steps=ddim_steps,
            denoise_steps=denoise_steps,
            eta=eta,
            mask_dilate_px=mask_dilate_px,
            mask_feather_px=mask_feather_px,
            seed=seed,
            H=256,
            W=256,
            model=_cached_model,  # Use cached model
            sampler=_cached_sampler,  # Use cached sampler
            cfg=_cached_cfg  # Use cached config
        )
        
        # Clean up temp files
        try:
            os.unlink(ref_path)
            os.unlink(mask_path)
        except:
            pass
        
        return output_path, f"Generation successful! Output path: {output_path}"
    
    except Exception as e:
        import traceback
        error_msg = f"Generation failed: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


# Create Gradio interface
def create_ui():
    with gr.Blocks(title="Iris Image Generation") as demo:
        gr.Markdown("# 🎨 Iris Image Generation Tool")
        gr.Markdown("Select a reference image and mask image, adjust the strength parameter to generate synthetic images.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Settings")
                
                ref_image = gr.Image(
                    label="Reference Image",
                    type="pil",
                    height=300
                )
                
                mask_image = gr.Image(
                    label="Mask Image",
                    type="pil",
                    height=300
                )
                
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="Strength - Controls the degree of change outside the iris region",
                    info="Higher values result in more changes outside the iris region"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    config_path = gr.Textbox(
                        label="Config Path",
                        value=DEFAULT_CONFIG,
                        placeholder="local_configs/latent-diffusion/iris_stageA_sota_64.yaml"
                    )
                    
                    ckpt_path = gr.Textbox(
                        label="Checkpoint Path",
                        value=DEFAULT_CKPT,
                        placeholder="models/best_ldm_model.ckpt"
                    )
                    
                    class_id = gr.Number(
                        label="Class ID",
                        value=DEFAULT_CLASS_ID,
                        precision=0
                    )
                
                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output Result")
                
                output_image = gr.Image(
                    label="Generated Image",
                    height=400
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    lines=3,
                    interactive=False
                )
        
        # Fixed parameters info
        with gr.Accordion("Fixed Parameters", open=False):
            gr.Markdown("""
            The following parameters are fixed (optimized for speed):
            - **ddim_steps**: 100
            - **denoise_steps**: 30 (reduced from 50 for faster generation)
            - **eta**: 0.0
            - **mask_dilate_px**: 2
            - **mask_feather_px**: 3
            - **seed**: 0
            
            **Performance Optimization Notes:**
            - The model will be loaded and cached on first use, subsequent generations will reuse the cached model, significantly reducing loading time
            - denoise_steps has been optimized to 30 to improve generation speed while maintaining quality
            """)
        
        # Connect button to function
        generate_btn.click(
            fn=generate_image,
            inputs=[ref_image, mask_image, strength, config_path, ckpt_path, class_id],
            outputs=[output_image, status_text]
        )
        
        gr.Markdown("---")
        gr.Markdown("### Usage Instructions")
        gr.Markdown("""
        1. Upload a reference image (original iris image)
        2. Upload a mask image (binary image, white regions indicate iris area)
        3. Adjust the **Strength** parameter (0-1, controls the degree of change outside the iris region)
        4. Set the model checkpoint path and config file path
        5. Click the "Generate Image" button
        6. The generated image will be displayed on the right
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,  # Auto-find available port
        share=False,
        theme=gr.themes.Soft()
    )

