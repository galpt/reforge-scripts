#!/usr/bin/env python3
"""
Auto CFG Optimizer
Dynamically adjusts CFG scale during generation for optimal balance between prompt adherence and speed

This script provides:
1. Real-time CFG adjustment during generation
2. Multiple transition curves (Linear, Exponential, Cosine, Sigmoid)
3. Configurable start/end values
4. Early step preservation
5. Debug logging and monitoring
"""

import modules.scripts as scripts
import gradio as gr
import math
from modules import script_callbacks, shared, processing
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
from typing import Any
import torch

class AutoCFGOptimizer:
    """Auto CFG optimization system"""
    
    def __init__(self):
        self.original_cfg = None
        self.current_step = 0
        self.total_steps = 0
        self.cfg_schedule = []
        self.debug_enabled = False
        self.log_enabled = True
        
    def calculate_cfg_schedule(self, original_cfg: float, total_steps: int, 
                              start_multiplier: float = 1.0, end_value: float = 0.0,
                              curve_type: str = "Cosine", preserve_early_pct: float = 20.0) -> list:
        """Calculate CFG values for each step"""
        schedule = []
        start_cfg = original_cfg * start_multiplier
        preserve_steps = int(total_steps * preserve_early_pct / 100.0)
        
        for step in range(total_steps):
            if step < preserve_steps:
                # Preserve original CFG for early steps
                cfg_value = start_cfg
            else:
                # Calculate transition progress (0 to 1)
                transition_step = step - preserve_steps
                transition_total = total_steps - preserve_steps
                progress = transition_step / max(transition_total - 1, 1)
                
                # Apply curve transformation
                if curve_type == "Linear":
                    factor = progress
                elif curve_type == "Exponential":
                    factor = progress ** 2
                elif curve_type == "Cosine":
                    factor = (1 - math.cos(progress * math.pi)) / 2
                elif curve_type == "Sigmoid":
                    # Sigmoid curve (S-shaped)
                    x = (progress - 0.5) * 12  # Scale for steeper curve
                    factor = 1 / (1 + math.exp(-x))
                else:
                    factor = progress
                
                # Interpolate between start and end CFG
                cfg_value = start_cfg * (1 - factor) + end_value * factor
            
            schedule.append(max(cfg_value, 0.0))  # Ensure non-negative
            
        return schedule
    
    def setup_cfg_hook(self, p, enabled: bool, start_multiplier: float, end_value: float,
                      curve_type: str, preserve_early_pct: float, debug: bool, log_changes: bool):
        """Setup CFG modification hook"""
        if not enabled:
            return
            
        self.original_cfg = p.cfg_scale
        self.total_steps = p.steps
        self.current_step = 0
        self.debug_enabled = debug
        self.log_enabled = log_changes
        
        # Calculate CFG schedule
        self.cfg_schedule = self.calculate_cfg_schedule(
            self.original_cfg, self.total_steps, start_multiplier, 
            end_value, curve_type, preserve_early_pct
        )
        
        if self.log_enabled:
            print(f"[Auto CFG] Initialized with {self.total_steps} steps, original CFG: {self.original_cfg}")
            print(f"[Auto CFG] Schedule: {[round(cfg, 2) for cfg in self.cfg_schedule[:10]]}{'...' if len(self.cfg_schedule) > 10 else ''}")
        
        # Hook into the sampling process
        self._setup_sampling_hook(p)
    
    def _setup_sampling_hook(self, p):
        """Setup CFG modification for Forge sampling"""
        try:
            # Reset step counter for each sampling
            self.current_step = 0
            
            # Create conditioning modifier function
            def cfg_conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
                if self.current_step < len(self.cfg_schedule):
                    new_cfg = self.cfg_schedule[self.current_step]
                    
                    if self.debug_enabled or self.log_enabled:
                        print(f"[Auto CFG] Step {self.current_step+1}/{self.total_steps}: CFG {cond_scale:.2f} → {new_cfg:.2f}")
                    
                    # Update the CFG scale
                    cond_scale = new_cfg
                else:
                    if self.debug_enabled or self.log_enabled:
                        print(f"[Auto CFG] Step {self.current_step+1}: Using original CFG {cond_scale:.2f} (beyond schedule)")
                    
                self.current_step += 1
                return model, x, timestep, uncond, cond, cond_scale, model_options, seed
            
            # Add conditioning modifier to UNet model options
            unet = p.sd_model.forge_objects.unet
            if 'conditioning_modifiers' not in unet.model_options:
                unet.model_options['conditioning_modifiers'] = []
            unet.model_options['conditioning_modifiers'].append(cfg_conditioning_modifier)
            
            if self.log_enabled:
                print(f"[Auto CFG] Successfully added conditioning modifier. Original CFG: {self.original_cfg}")
                print(f"[Auto CFG] CFG Schedule: {[round(cfg, 2) for cfg in self.cfg_schedule[:5]]}{'...' if len(self.cfg_schedule) > 5 else ''}")
                
        except Exception as e:
            if self.log_enabled:
                print(f"[Auto CFG] Warning: Could not setup CFG modification: {e}")
                import traceback
                traceback.print_exc()

class AutoCFGOptimizerScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.cfg_optimizer = AutoCFGOptimizer()
        self.alwayson = True  # Make this an AlwaysVisible script

    def title(self):
        return "Auto CFG Optimizer"

    def describe(self):
        return "Dynamically adjusts CFG scale during generation for optimal balance between prompt adherence and speed"

    def show(self, is_img2img):
        return scripts.AlwaysVisible  # Always visible script

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Auto CFG Optimizer Settings", open=False):
                enabled = gr.Checkbox(label="Enable Auto CFG Optimizer", value=False)
                
                with gr.Row():
                    cfg_start_multiplier = gr.Slider(
                        minimum=0.5, 
                        maximum=2.0, 
                        step=0.1, 
                        label="CFG Start Multiplier", 
                        value=1.0,
                        info="Multiplier for initial CFG value (1.0 = use original CFG)"
                    )
                    
                    cfg_end_value = gr.Slider(
                        minimum=0.0, 
                        maximum=5.0, 
                        step=0.1, 
                        label="CFG End Value", 
                        value=0.0,
                        info="CFG value for the final step"
                    )
                
                with gr.Row():
                    transition_curve = gr.Dropdown(
                        label="Transition Curve", 
                        choices=["Linear", "Exponential", "Cosine", "Sigmoid"],
                        value="Cosine",
                        info="How CFG decreases over time"
                    )
                    
                    preserve_early_steps = gr.Slider(
                        minimum=0, 
                        maximum=50, 
                        step=1, 
                        label="Preserve Early Steps (%)", 
                        value=20,
                        info="Percentage of steps to keep original CFG"
                    )
                
                with gr.Row():
                    show_debug = gr.Checkbox(label="Show Debug Info", value=False)
                    log_cfg_changes = gr.Checkbox(label="Log CFG Changes", value=True)
        
        return [enabled, cfg_start_multiplier, cfg_end_value, transition_curve, 
                preserve_early_steps, show_debug, log_cfg_changes]

    # AlwaysVisible scripts don't use run() method - they use process() instead
    
    def process(self, p, *args):
        """Called before processing begins for AlwaysVisible scripts"""
        # Get the enabled state from the UI arguments
        if len(args) >= 1:
            enabled = args[0]
            if enabled:
                try:
                    # Get all the parameters from args
                    cfg_start_multiplier = args[1] if len(args) > 1 else 1.0
                    cfg_end_value = args[2] if len(args) > 2 else 0.0
                    transition_curve = args[3] if len(args) > 3 else "Cosine"
                    preserve_early_steps = args[4] if len(args) > 4 else 20
                    show_debug = args[5] if len(args) > 5 else False
                    log_cfg_changes = args[6] if len(args) > 6 else True
                    
                    # Setup CFG optimization
                    self.cfg_optimizer.setup_cfg_hook(
                        p, enabled, cfg_start_multiplier, cfg_end_value,
                        transition_curve, preserve_early_steps, show_debug, log_cfg_changes
                    )
                    
                    if log_cfg_changes:
                        print(f"[Auto CFG] Setup completed. Original CFG: {self.cfg_optimizer.original_cfg}, Target end CFG: {cfg_end_value}")
                except Exception as e:
                    print(f"[Auto CFG] Error in process: {e}")
                    import traceback
                    traceback.print_exc()

# Script class is automatically detected by WebUI