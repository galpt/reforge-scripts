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
6. Fully automatic mode with intelligent prompt analysis
7. Real-time parameter awareness and adaptive optimization
"""

import modules.scripts as scripts
import gradio as gr
import math
import re
from modules import script_callbacks, shared, processing
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
from typing import Any, Dict, List, Tuple
import torch

class AutoCFGOptimizer:
    """Auto CFG optimization system with intelligent analysis"""
    
    def __init__(self):
        self.original_cfg = None
        self.current_step = 0
        self.total_steps = 0
        self.cfg_schedule = []
        self.debug_enabled = False
        self.log_enabled = True
        self.auto_mode = False
        self.prompt_analysis = {}
        self.generation_context = {}
        
        # Auto mode intelligence parameters
        self.complexity_weights = {
            'prompt_length': 0.15,
            'detail_keywords': 0.25,
            'style_keywords': 0.20,
            'quality_keywords': 0.15,
            'negative_strength': 0.25
        }
        
        # Keyword databases for analysis
        self.detail_keywords = {
            'high': ['detailed', 'intricate', 'complex', 'elaborate', 'ornate', 'sophisticated', 
                    'fine details', 'highly detailed', 'ultra detailed', 'extremely detailed'],
            'medium': ['clear', 'sharp', 'defined', 'crisp', 'focused', 'precise'],
            'low': ['simple', 'minimal', 'basic', 'clean', 'plain']
        }
        
        self.style_keywords = {
            'realistic': ['photorealistic', 'realistic', 'photograph', 'photo', 'real', 'lifelike'],
            'artistic': ['painting', 'artwork', 'artistic', 'stylized', 'illustration', 'drawn'],
            'anime': ['anime', 'manga', 'cartoon', 'animated', '2d'],
            'abstract': ['abstract', 'surreal', 'conceptual', 'experimental']
        }
        
        self.quality_keywords = {
            'ultra': ['masterpiece', 'best quality', 'ultra quality', 'perfect', 'flawless'],
            'high': ['high quality', 'good quality', 'professional', 'premium'],
            'standard': ['quality', 'decent', 'normal']
        }
        
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
    
    def analyze_prompt_complexity(self, positive_prompt: str, negative_prompt: str = "") -> Dict[str, float]:
        """Analyze prompt complexity with advanced fallback mechanisms for edge cases"""
        analysis = {
            'prompt_length_score': 0.0,
            'detail_score': 0.0,
            'style_score': 0.0,
            'quality_score': 0.0,
            'negative_strength': 0.0,
            'linguistic_complexity': 0.0,
            'structural_complexity': 0.0,
            'semantic_density': 0.0,
            'overall_complexity': 0.0
        }
        
        # Normalize prompts for analysis
        pos_lower = positive_prompt.lower()
        neg_lower = negative_prompt.lower()
        words = positive_prompt.split()
        
        # 1. Enhanced prompt length analysis with non-linear scaling
        word_count = len(words)
        analysis['prompt_length_score'] = min(word_count / 50.0, 1.0)  # Normalize to 0-1
        
        # 2. Detail keyword analysis with fuzzy matching fallback
        detail_score = 0.0
        detail_matches = 0
        for level, keywords in self.detail_keywords.items():
            weight = {'high': 1.0, 'medium': 0.6, 'low': 0.2}[level]
            for keyword in keywords:
                if keyword in pos_lower:
                    detail_score += weight
                    detail_matches += 1
        
        # Fallback: Linguistic pattern analysis for detail indicators
        if detail_matches == 0:
            detail_score += self._analyze_detail_patterns(pos_lower)
        
        analysis['detail_score'] = min(detail_score / 3.0, 1.0)  # Normalize
        
        # 3. Style keyword analysis with semantic fallbacks
        style_score = 0.0
        style_matches = 0
        style_weights = {'realistic': 0.8, 'artistic': 0.6, 'anime': 0.4, 'abstract': 0.3}
        for style, keywords in self.style_keywords.items():
            for keyword in keywords:
                if keyword in pos_lower:
                    style_score = max(style_score, style_weights[style])
                    style_matches += 1
        
        # Fallback: Infer style from linguistic patterns
        if style_matches == 0:
            style_score = self._infer_style_from_patterns(pos_lower)
        
        analysis['style_score'] = style_score
        
        # 4. Quality keyword analysis with intensity fallback
        quality_score = 0.0
        quality_matches = 0
        for level, keywords in self.quality_keywords.items():
            weight = {'ultra': 1.0, 'high': 0.7, 'standard': 0.4}[level]
            for keyword in keywords:
                if keyword in pos_lower:
                    quality_score = max(quality_score, weight)
                    quality_matches += 1
        
        # Fallback: Analyze adjective intensity and descriptive density
        if quality_matches == 0:
            quality_score = self._analyze_quality_indicators(pos_lower)
        
        analysis['quality_score'] = quality_score
        
        # 5. Enhanced negative prompt analysis
        neg_word_count = len(negative_prompt.split())
        analysis['negative_strength'] = min(neg_word_count / 30.0, 1.0)
        
        # 6. NEW: Linguistic complexity analysis (fallback for unknown content)
        analysis['linguistic_complexity'] = self._analyze_linguistic_complexity(words)
        
        # 7. NEW: Structural complexity (punctuation, formatting, special tokens)
        analysis['structural_complexity'] = self._analyze_structural_complexity(positive_prompt)
        
        # 8. NEW: Semantic density (concept richness per word)
        analysis['semantic_density'] = self._analyze_semantic_density(words)
        
        # Enhanced overall complexity calculation with fallback weighting
        keyword_coverage = (detail_matches + style_matches + quality_matches) / 10.0  # Normalize
        
        if keyword_coverage > 0.3:  # Good keyword coverage - use standard weighting
            overall = (
                analysis['prompt_length_score'] * 0.15 +
                analysis['detail_score'] * 0.25 +
                analysis['style_score'] * 0.20 +
                analysis['quality_score'] * 0.15 +
                analysis['negative_strength'] * 0.25
            )
        else:  # Poor keyword coverage - rely more on linguistic analysis
            overall = (
                analysis['prompt_length_score'] * 0.10 +
                analysis['detail_score'] * 0.15 +
                analysis['style_score'] * 0.10 +
                analysis['quality_score'] * 0.10 +
                analysis['negative_strength'] * 0.15 +
                analysis['linguistic_complexity'] * 0.20 +
                analysis['structural_complexity'] * 0.10 +
                analysis['semantic_density'] * 0.10
            )
        
        analysis['overall_complexity'] = min(overall, 1.0)
        return analysis
    
    def _analyze_detail_patterns(self, text: str) -> float:
        """Fallback: Analyze linguistic patterns that suggest detail requirements"""
        detail_indicators = [
            # Adjective patterns
            len(re.findall(r'\b\w+ly\b', text)) * 0.1,  # Adverbs often indicate detail
            len(re.findall(r'\b\w{8,}\b', text)) * 0.05,  # Long words often descriptive
            # Descriptive patterns
            text.count(',') * 0.02,  # Comma-separated details
            text.count('with') * 0.05,  # "with X" patterns
            text.count('featuring') * 0.1,  # Feature descriptions
            # Technical terms
            len(re.findall(r'\d+k|\d+p|hdr|uhd|4k|8k', text)) * 0.2,  # Resolution terms
        ]
        return min(sum(detail_indicators), 1.0)
    
    def _infer_style_from_patterns(self, text: str) -> float:
        """Fallback: Infer artistic style from linguistic patterns"""
        # Realistic indicators
        realistic_patterns = len(re.findall(r'\b(shot|camera|lens|lighting|shadow|highlight)\b', text))
        if realistic_patterns > 0:
            return 0.8
        
        # Artistic indicators
        artistic_patterns = len(re.findall(r'\b(brush|stroke|canvas|paint|color|palette)\b', text))
        if artistic_patterns > 0:
            return 0.6
        
        # Character/anime indicators
        character_patterns = len(re.findall(r'\b(character|girl|boy|person|face|eyes|hair)\b', text))
        if character_patterns > 2:
            return 0.4
        
        # Default to medium complexity for unknown styles
        return 0.5
    
    def _analyze_quality_indicators(self, text: str) -> float:
        """Fallback: Analyze quality intent from linguistic patterns"""
        quality_indicators = [
            # Superlatives and intensifiers
            len(re.findall(r'\b(best|perfect|amazing|stunning|incredible|beautiful)\b', text)) * 0.2,
            len(re.findall(r'\b(ultra|super|hyper|extremely|highly)\b', text)) * 0.15,
            # Professional terms
            len(re.findall(r'\b(professional|studio|premium|deluxe|luxury)\b', text)) * 0.25,
            # Technical quality terms
            len(re.findall(r'\b(sharp|crisp|clear|defined|precise)\b', text)) * 0.1,
        ]
        return min(sum(quality_indicators), 1.0)
    
    def _analyze_linguistic_complexity(self, words: List[str]) -> float:
        """Analyze linguistic complexity for unknown content"""
        if not words:
            return 0.0
        
        # Average word length (longer words often more complex)
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = min(avg_word_length / 8.0, 1.0)  # Normalize to 8-char average
        
        # Vocabulary diversity (unique words / total words)
        diversity_score = len(set(word.lower() for word in words)) / len(words)
        
        # Compound word detection
        compound_score = len([w for w in words if len(w) > 10]) / max(len(words), 1)
        
        return min((length_score * 0.4 + diversity_score * 0.4 + compound_score * 0.2), 1.0)
    
    def _analyze_structural_complexity(self, text: str) -> float:
        """Analyze structural complexity from formatting and punctuation"""
        structure_indicators = [
            # Punctuation complexity
            text.count('(') * 0.05,  # Parenthetical additions
            text.count('[') * 0.05,  # Bracket notation
            text.count(':') * 0.03,  # Colons for descriptions
            text.count(';') * 0.04,  # Semicolons for complex lists
            # Special tokens
            text.count('<') * 0.02,  # Lora/embedding tokens
            text.count('{') * 0.02,  # Attention modification
            # Line breaks and formatting
            text.count('\n') * 0.1,  # Multi-line prompts
        ]
        return min(sum(structure_indicators), 1.0)
    
    def _analyze_semantic_density(self, words: List[str]) -> float:
        """Analyze semantic density - concept richness per word"""
        if not words:
            return 0.0
        
        # Noun density (nouns often carry semantic weight)
        potential_nouns = [w for w in words if len(w) > 3 and not w.lower() in 
                          ['the', 'and', 'with', 'from', 'that', 'this', 'have', 'been', 'were']]
        noun_density = len(potential_nouns) / len(words)
        
        # Adjective density (descriptive richness)
        potential_adjectives = [w for w in words if w.endswith(('ed', 'ing', 'ful', 'ous', 'ive'))]
        adj_density = len(potential_adjectives) / len(words)
        
        # Concept clustering (repeated semantic fields)
        word_stems = [w[:4] for w in words if len(w) > 4]
        clustering = 1.0 - (len(set(word_stems)) / max(len(word_stems), 1))
        
        return min((noun_density * 0.4 + adj_density * 0.3 + clustering * 0.3), 1.0)
    
    def calculate_auto_cfg_params(self, p, complexity_analysis: Dict[str, float]) -> Tuple[float, float, str, float]:
        """Calculate optimal CFG parameters with advanced fallback-aware analysis"""
        complexity = complexity_analysis['overall_complexity']
        original_cfg = p.cfg_scale
        steps = p.steps
        
        # Extract individual complexity components for nuanced decisions
        linguistic_complexity = complexity_analysis.get('linguistic_complexity', 0.0)
        structural_complexity = complexity_analysis.get('structural_complexity', 0.0)
        semantic_density = complexity_analysis.get('semantic_density', 0.0)
        style_score = complexity_analysis['style_score']
        detail_score = complexity_analysis['detail_score']
        quality_score = complexity_analysis['quality_score']
        
        # Base parameters that work well for most cases
        base_start_multiplier = 1.0
        base_end_value = 1.0
        base_preserve_pct = 20.0
        
        # Enhanced complexity-based adjustments with fallback considerations
        if complexity < 0.3:  # Low complexity
            start_multiplier = base_start_multiplier * 0.9
            end_value = max(original_cfg * 0.3, 1.0)
            curve_type = "Linear"
            preserve_pct = 15.0
            
            # Fallback adjustments for low-keyword-match scenarios
            if linguistic_complexity > 0.6:  # Complex language despite low keyword match
                start_multiplier *= 1.1
                end_value *= 1.3
                preserve_pct += 5.0
                
        elif complexity < 0.6:  # Medium complexity
            start_multiplier = base_start_multiplier
            end_value = max(original_cfg * 0.4, 1.5)
            curve_type = "Cosine"
            preserve_pct = 20.0
            
            # Structural complexity adjustments
            if structural_complexity > 0.5:  # Complex formatting suggests detailed requirements
                curve_type = "Sigmoid"  # More gradual for complex structures
                preserve_pct += 5.0
                
        elif complexity < 0.8:  # High complexity
            start_multiplier = base_start_multiplier * 1.1
            end_value = max(original_cfg * 0.5, 2.0)
            curve_type = "Cosine"
            preserve_pct = 25.0
            
            # Semantic density adjustments
            if semantic_density > 0.7:  # High concept density
                start_multiplier *= 1.15
                end_value *= 1.1
                preserve_pct += 7.0
                
        else:  # Very high complexity
            start_multiplier = base_start_multiplier * 1.2
            end_value = max(original_cfg * 0.6, 2.5)
            curve_type = "Sigmoid"
            preserve_pct = 30.0
            
            # Maximum complexity fallback adjustments
            if linguistic_complexity > 0.8 or semantic_density > 0.8:
                start_multiplier *= 1.1  # Even higher start for very complex language
                preserve_pct += 10.0  # More preservation for complex content
        
        # Advanced step count adjustments with complexity consideration
        if steps < 20:
            preserve_pct *= 0.7
            # For few steps with high complexity, be more conservative
            if complexity > 0.7:
                preserve_pct *= 1.2
        elif steps > 50:
            preserve_pct *= 1.2
            # For many steps, can afford more aggressive reduction if complexity allows
            if complexity < 0.4 and linguistic_complexity < 0.5:
                end_value *= 0.9  # More aggressive reduction for simple content
        
        # Enhanced style-specific adjustments
        if style_score > 0.7:  # Realistic styles
            start_multiplier *= 1.1
            end_value *= 1.2
        elif style_score > 0.0 and style_score <= 0.4:  # Anime/abstract styles
            # These styles often work well with lower CFG
            start_multiplier *= 0.95
            end_value *= 0.9
        
        # Quality-driven adjustments
        if quality_score > 0.8:  # High quality requirements
            preserve_pct += 8.0
            start_multiplier *= 1.05
        elif quality_score == 0.0 and linguistic_complexity > 0.6:
            # No quality keywords but complex language - infer quality intent
            preserve_pct += 5.0
        
        # Detail-driven adjustments
        if detail_score > 0.8:  # High detail requirements
            curve_type = "Sigmoid"  # More gradual for detailed content
            preserve_pct += 10.0
        elif detail_score == 0.0 and structural_complexity > 0.5:
            # No detail keywords but complex structure - infer detail intent
            preserve_pct += 5.0
            curve_type = "Cosine"  # Balanced approach
        
        # Fallback safety nets for edge cases
        if all(score < 0.1 for score in [style_score, detail_score, quality_score]):
            # Very low keyword matching - rely heavily on linguistic analysis
            if linguistic_complexity > 0.7:
                # Complex language suggests sophisticated intent
                start_multiplier = base_start_multiplier * 1.1
                end_value = max(original_cfg * 0.5, 2.0)
                curve_type = "Cosine"
                preserve_pct = 25.0
            elif semantic_density > 0.6:
                # High concept density suggests detailed requirements
                start_multiplier = base_start_multiplier
                end_value = max(original_cfg * 0.4, 1.5)
                curve_type = "Cosine"
                preserve_pct = 22.0
            else:
                # Truly simple content - conservative approach
                start_multiplier = base_start_multiplier * 0.95
                end_value = max(original_cfg * 0.35, 1.2)
                curve_type = "Linear"
                preserve_pct = 18.0
        
        return start_multiplier, end_value, curve_type, min(preserve_pct, 45.0)
    
    def extract_generation_context(self, p) -> Dict[str, Any]:
        """Extract all relevant generation parameters for analysis"""
        context = {
            'positive_prompt': getattr(p, 'prompt', ''),
            'negative_prompt': getattr(p, 'negative_prompt', ''),
            'steps': getattr(p, 'steps', 20),
            'cfg_scale': getattr(p, 'cfg_scale', 7.0),
            'width': getattr(p, 'width', 512),
            'height': getattr(p, 'height', 512),
            'sampler_name': getattr(p, 'sampler_name', 'Unknown'),
            'scheduler': getattr(p, 'scheduler', 'Unknown'),
            'seed': getattr(p, 'seed', -1),
            'batch_size': getattr(p, 'batch_size', 1),
            'n_iter': getattr(p, 'n_iter', 1)
        }
        
        # Calculate resolution factor (higher res might need different CFG)
        total_pixels = context['width'] * context['height']
        context['resolution_factor'] = total_pixels / (512 * 512)  # Normalize to 512x512
        
        return context
    
    def setup_cfg_hook(self, p, enabled: bool, auto_mode: bool = False, start_multiplier: float = 1.0, 
                      end_value: float = 0.0, curve_type: str = "Cosine", preserve_early_pct: float = 20.0, 
                      debug: bool = False, log_changes: bool = True):
        """Setup CFG modification hook with automatic or manual mode"""
        if not enabled:
            return
            
        self.original_cfg = p.cfg_scale
        self.total_steps = p.steps
        self.current_step = 0
        self.debug_enabled = debug
        self.log_enabled = log_changes
        self.auto_mode = auto_mode
        
        if auto_mode:
            # Extract generation context for intelligent analysis
            self.generation_context = self.extract_generation_context(p)
            
            # Analyze prompt complexity
            self.prompt_analysis = self.analyze_prompt_complexity(
                self.generation_context['positive_prompt'],
                self.generation_context['negative_prompt']
            )
            
            # Calculate optimal parameters automatically
            start_multiplier, end_value, curve_type, preserve_early_pct = self.calculate_auto_cfg_params(
                p, self.prompt_analysis
            )
            
            if self.log_enabled:
                print(f"[Auto CFG] Automatic mode enabled - Analyzing generation context...")
                print(f"[Auto CFG] Prompt complexity: {self.prompt_analysis['overall_complexity']:.2f}")
                print(f"[Auto CFG] Auto-calculated params: start_mult={start_multiplier:.2f}, end={end_value:.2f}, curve={curve_type}, preserve={preserve_early_pct:.1f}%")
        
        # Calculate CFG schedule
        self.cfg_schedule = self.calculate_cfg_schedule(
            self.original_cfg, self.total_steps, start_multiplier, 
            end_value, curve_type, preserve_early_pct
        )
        
        if self.log_enabled:
            mode_str = "Automatic" if auto_mode else "Manual"
            print(f"[Auto CFG] {mode_str} mode initialized with {self.total_steps} steps, original CFG: {self.original_cfg}")
            print(f"[Auto CFG] Schedule: {[round(cfg, 2) for cfg in self.cfg_schedule[:10]]}{'...' if len(self.cfg_schedule) > 10 else ''}")
            
            if auto_mode and self.debug_enabled:
                print(f"[Auto CFG] Detailed analysis:")
                for key, value in self.prompt_analysis.items():
                    if key != 'overall_complexity':
                        print(f"  - {key}: {value:.3f}")
                print(f"[Auto CFG] Generation context: {self.generation_context['width']}x{self.generation_context['height']}, {self.generation_context['sampler_name']}")
        
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
                
                # Mode selection
                with gr.Row():
                    auto_mode = gr.Checkbox(
                        label="🤖 Automatic Mode", 
                        value=True,
                        info="Let AI analyze your prompt and automatically optimize CFG parameters"
                    )
                
                # Manual configuration (shown when auto mode is disabled)
                with gr.Group() as manual_group:
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
                
                # Advanced options
                with gr.Row():
                    show_debug = gr.Checkbox(label="Show Debug Info", value=False)
                    log_cfg_changes = gr.Checkbox(label="Log CFG Changes", value=True)
                
                # Auto mode info display
                with gr.Group() as auto_info:
                    gr.Markdown("""
                    **🤖 Automatic Mode Features:**
                    - **Intelligent Prompt Analysis**: Analyzes prompt complexity, detail level, style, and quality keywords
                    - **Real-time Parameter Optimization**: Automatically calculates optimal CFG start/end values and transition curves
                    - **Context Awareness**: Considers resolution, steps, sampler, and other generation parameters
                    - **Style-Specific Optimization**: Adapts CFG strategy based on detected art style (realistic, anime, artistic, etc.)
                    - **Quality-Driven Adjustments**: Higher CFG preservation for quality-focused prompts
                    
                    Simply enable Auto CFG Optimizer and check Automatic Mode - the AI will handle the rest!
                    """)
                
                # JavaScript to show/hide manual controls based on auto mode
                auto_mode.change(
                    fn=None,
                    inputs=[auto_mode],
                    outputs=[manual_group, auto_info],
                    _js="""(auto_mode) => {
                        return [auto_mode ? {visible: false} : {visible: true}, 
                                auto_mode ? {visible: true} : {visible: false}];
                    }"""
                )
        
        return [enabled, auto_mode, cfg_start_multiplier, cfg_end_value, transition_curve, 
                preserve_early_steps, show_debug, log_cfg_changes]

    # AlwaysVisible scripts don't use run() method - they use process() instead
    
    def process(self, p, *args):
        """Called before processing begins for AlwaysVisible scripts"""
        # Get the enabled state from the UI arguments
        if len(args) >= 1:
            enabled = args[0]
            if enabled:
                try:
                    # Get all the parameters from args (updated order with auto_mode)
                    auto_mode = args[1] if len(args) > 1 else True
                    cfg_start_multiplier = args[2] if len(args) > 2 else 1.0
                    cfg_end_value = args[3] if len(args) > 3 else 0.0
                    transition_curve = args[4] if len(args) > 4 else "Cosine"
                    preserve_early_steps = args[5] if len(args) > 5 else 20
                    show_debug = args[6] if len(args) > 6 else False
                    log_cfg_changes = args[7] if len(args) > 7 else True
                    
                    # Setup CFG optimization with automatic or manual mode
                    self.cfg_optimizer.setup_cfg_hook(
                        p, enabled, auto_mode, cfg_start_multiplier, cfg_end_value,
                        transition_curve, preserve_early_steps, show_debug, log_cfg_changes
                    )
                    
                    if log_cfg_changes:
                        mode_str = "Automatic" if auto_mode else "Manual"
                        if auto_mode:
                            print(f"[Auto CFG] {mode_str} mode setup completed. AI will analyze and optimize CFG automatically.")
                        else:
                            print(f"[Auto CFG] {mode_str} mode setup completed. Original CFG: {self.cfg_optimizer.original_cfg}, Target end CFG: {cfg_end_value}")
                except Exception as e:
                    print(f"[Auto CFG] Error in process: {e}")
                    import traceback
                    traceback.print_exc()

# Script class is automatically detected by WebUI