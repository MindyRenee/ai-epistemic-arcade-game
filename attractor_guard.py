"""
Attractor-Basin Trajectory Guard (May 2026)
============================================
Based on findings from:
  - Akarlar & Varshney, "Hallucination as Trajectory Commitment" (arXiv:2604.15400, Apr 2026)
  - Cherukuri & Varshney, "Hallucination Basins" (arXiv:2604.04743, Apr 2026)
  - [May 2026 geometric perspective on next-token prediction]

Key findings incorporated:
  1. Linear projection of hallucination-correlated directions from residual streams
     does NOT prevent hallucination (model reconstructs signal within 2-3 steps).
  2. Hallucination is an asymmetric attractor basin: easy to enter, hard to escape.
  3. The prompt encoding (step-0) already commits the model to a basin regime.
  4. Effective intervention requires:
     a) Regime-aware detection at prompt time.
     b) Adaptive risk-aware steering proportional to basin proximity.
     c) Sustained multi-step window patching for correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class _LazyImports:
    _torch = None
    _transformers = None

    @classmethod
    def torch(cls):
        if cls._torch is None:
            import torch
            cls._torch = torch
        return cls._torch

    @classmethod
    def transformers(cls):
        if cls._transformers is None:
            import transformers
            cls._transformers = transformers
        return cls._transformers


@dataclass
class BasinGuardConfig:
    # Layer to hook for intervention (penultimate is common sweet spot)
    target_layer: int = -2

    # Number of steps for sustained window patching (single-step fails)
    patch_window_size: int = 4

    # Prompt-time regime detection: probe layer for step-0 classification
    regime_probe_layer: int = 15

    # Risk threshold for triggering intervention
    risk_threshold: float = 0.5

    # Maximum steering intensity
    lambda_max: float = 1.0

    # Record per-token diagnostics
    record_diagnostics: bool = True


@dataclass
class BasinDiagnostic:
    position: int
    token_str: str
    token_id: int
    radial_distance: float
    contraction_ratio: float
    risk_score: float
    was_patched: bool


class AttractorBasinGuard:
    """
    Implements May 2026 basin-aware trajectory control.

    Phase 1 – REGIME DETECTION (prompt time):
        Extract step-0 hidden state at regime_probe_layer.
        Compare to factual vs hallucinated reference states.
        Estimate per-prompt hallucination risk.

    Phase 2 – ADAPTIVE STEERING (generation):
        If risk is high, compute dynamic steering intensity λ based on
        geometric features: radial distance to basin center and local
        contraction ratio.
        Apply sustained multi-step window patching.

    Phase 3 – MULTI-STEP CORRECTION:
        If trajectory is already in hallucination basin, apply coordinated
        activation patching across a window of steps to push across the
        separatrix into the factual basin.
    """

    def __init__(
        self,
        model,
        tokenizer,
        factual_trajectory: Sequence[str],
        hallucinated_examples: Sequence[str],
        config: Optional[BasinGuardConfig] = None,
    ):
        self.cfg = config or BasinGuardConfig()
        self.model = model
        self.tokenizer = tokenizer
        self._torch = _LazyImports.torch()

        # Build reference geometries from the model's own hidden states
        self.factual_centroid: Dict[int, np.ndarray] = {}
        self.hallucinated_centroid: Dict[int, np.ndarray] = {}
        self.steering_vector: Dict[int, np.ndarray] = {}
        self._build_reference_states(factual_trajectory, hallucinated_examples)

        self.diagnostics: List[BasinDiagnostic] = []
        self._token_count = 0
        self._patch_remaining = 0
        self._risk_at_prompt = 0.0
        self._hook_handle = self._register_hook()

    def _build_reference_states(
        self, factual: Sequence[str], hallucinated: Sequence[str]
    ) -> None:
        torch = self._torch
        device = next(self.model.parameters()).device
        layers_to_probe = [self.cfg.regime_probe_layer, self.cfg.target_layer]

        self.model.eval()
        with torch.no_grad():
            for layer_idx in layers_to_probe:
                factual_vecs: List[np.ndarray] = []
                for txt in factual:
                    inputs = self.tokenizer(txt, return_tensors="pt").to(device)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    h = outputs.hidden_states[layer_idx]
                    # Use final token as the reference state for this example
                    last = h[0, -1, :].cpu().numpy()
                    factual_vecs.append(last)

                hall_vecs: List[np.ndarray] = []
                for txt in hallucinated:
                    inputs = self.tokenizer(txt, return_tensors="pt").to(device)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    h = outputs.hidden_states[layer_idx]
                    last = h[0, -1, :].cpu().numpy()
                    hall_vecs.append(last)

                f_cent = np.mean(factual_vecs, axis=0)
                h_cent = np.mean(hall_vecs, axis=0)
                self.factual_centroid[layer_idx] = f_cent
                self.hallucinated_centroid[layer_idx] = h_cent
                # Steering direction: push away from hallucination, toward factual
                v = f_cent - h_cent
                norm = np.linalg.norm(v)
                if norm > 1e-8:
                    v = v / norm
                self.steering_vector[layer_idx] = v

    # ------------------------------------------------------------------ #
    # Prompt-time regime detection
    # ------------------------------------------------------------------ #

    def assess_prompt_risk(self, prompt: str) -> float:
        """
        Step-0 regime encoding assessment.
        Returns risk score in [0,1] based on proximity to hallucination basin.
        """
        torch = self._torch
        device = next(self.model.parameters()).device
        layer = self.cfg.regime_probe_layer

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            outputs = self.model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[layer]
            h0 = h[0, -1, :].cpu().numpy()

        f_cent = self.factual_centroid[layer]
        h_cent = self.hallucinated_centroid[layer]

        d_factual = np.linalg.norm(h0 - f_cent)
        d_hall = np.linalg.norm(h0 - h_cent)

        # Risk = proximity to hallucination relative to total separation
        if d_factual + d_hall < 1e-8:
            risk = 0.0
        else:
            risk = d_hall / (d_factual + d_hall)

        self._risk_at_prompt = risk
        return risk

    # ------------------------------------------------------------------ #
    # Generation-time adaptive steering hook
    # ------------------------------------------------------------------ #

    def _steering_hook(self, module, _input, output):
        torch = self._torch
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        last_idx = hidden.shape[1] - 1
        h_last = hidden[:, last_idx, :]
        h_np = h_last.detach().cpu().numpy()

        layer = self.cfg.target_layer
        f_cent = self.factual_centroid[layer]
        h_cent = self.hallucinated_centroid[layer]
        v_steer = self.steering_vector[layer]

        # Geometric features for adaptive intensity
        radial_dist = float(np.linalg.norm(h_np - h_cent))
        radial_to_factual = float(np.linalg.norm(h_np - f_cent))

        # Contraction ratio: how fast is state collapsing into basin?
        # Approximate from running diagnostics
        contraction = self._estimate_contraction(h_np, h_cent)

        # Risk-aware steering intensity
        # Higher risk when: close to hallucination center AND contracting
        risk = self._compute_risk(radial_dist, contraction, radial_to_factual)

        # Determine if we should patch this step
        should_patch = False
        if risk > self.cfg.risk_threshold:
            should_patch = True
            self._patch_remaining = self.cfg.patch_window_size

        if self._patch_remaining > 0:
            should_patch = True
            self._patch_remaining -= 1

        was_patched = False
        if should_patch:
            # Adaptive intensity: proportional to risk, capped at lambda_max
            lam = min(risk * 2.0, self.cfg.lambda_max)

            # Apply steering: push toward factual, away from hallucination
            # Using the centroid-difference direction
            correction = lam * v_steer * np.linalg.norm(h_np)
            corrected = h_np + correction

            # Clamp to reasonable bounds to avoid exploding activations
            corrected = np.clip(corrected, -100.0, 100.0)

            corrected_t = torch.from_numpy(corrected).to(
                dtype=h_last.dtype, device=h_last.device
            )
            hidden[:, last_idx, :] = corrected_t
            was_patched = True

        if self.cfg.record_diagnostics:
            for b in range(h_np.shape[0]):
                self.diagnostics.append(
                    BasinDiagnostic(
                        position=self._token_count,
                        token_str="",
                        token_id=-1,
                        radial_distance=radial_dist,
                        contraction_ratio=contraction,
                        risk_score=risk,
                        was_patched=was_patched,
                    )
                )
            self._token_count += 1

        if is_tuple:
            return ((hidden,) + output[1:])
        return hidden

    def _estimate_contraction(
        self, h_np: np.ndarray, h_cent: np.ndarray
    ) -> float:
        """
        Estimate local contraction ratio based on recent diagnostics.
        If radial distance is decreasing rapidly, we're collapsing into basin.
        """
        if len(self.diagnostics) < 2:
            return 1.0
        recent = self.diagnostics[-5:]
        if len(recent) < 2:
            return 1.0
        distances = [d.radial_distance for d in recent]
        # Simple trend: ratio of most recent to average of earlier
        if len(distances) >= 3:
            recent_avg = np.mean(distances[-2:])
            earlier_avg = np.mean(distances[:-2]) if len(distances) > 2 else distances[0]
            if earlier_avg > 1e-8:
                return recent_avg / earlier_avg
        return 1.0

    def _compute_risk(
        self, radial_dist: float, contraction: float, dist_to_factual: float
    ) -> float:
        """
        Compute risk score from geometric features.
        Risk is high when close to hallucination center AND contracting.
        """
        # Normalize distances (heuristic: typical hidden state norms ~10-50)
        norm_radial = np.exp(-radial_dist / 20.0)
        norm_contraction = 1.0 - min(contraction, 1.0)
        norm_factual = np.exp(-dist_to_factual / 20.0)

        # Risk combines: proximity to hallucination, contraction, and distance from factual
        risk = 0.5 * norm_radial + 0.3 * norm_contraction - 0.2 * norm_factual
        return float(np.clip(risk, 0.0, 1.0))

    def _register_hook(self):
        layer_module = self._resolve_layer(self.cfg.target_layer)
        return layer_module.register_forward_hook(self._steering_hook)

    def _resolve_layer(self, layer_idx: int):
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers[layer_idx]
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h[layer_idx]
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers[layer_idx]
        if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
            return m.model.decoder.layers[layer_idx]
        raise RuntimeError(f"Cannot locate layer {layer_idx} in {type(m)}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, max_new_tokens: int = 40, **gen_kwargs) -> str:
        self.diagnostics.clear()
        self._token_count = 0
        self._patch_remaining = 0

        # Phase 1: Prompt-time regime detection
        risk = self.assess_prompt_risk(prompt)
        print(f"[AttractorGuard] Prompt risk: {risk:.3f}")

        # Pre-arm patch window if high risk detected at prompt time
        if risk > self.cfg.risk_threshold:
            self._patch_remaining = self.cfg.patch_window_size
            print(f"[AttractorGuard] Arming {self.cfg.patch_window_size}-step patch window")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )

        full_ids = output_ids[0].cpu().tolist()
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = full_ids[prompt_len:]

        for diag, tid in zip(self.diagnostics, generated_ids):
            diag.token_id = tid
            diag.token_str = self.tokenizer.decode([tid])

        return self.tokenizer.decode(full_ids, skip_special_tokens=True)

    def remove_hook(self) -> None:
        self._hook_handle.remove()

    def summary(self) -> str:
        lines = ["Attractor-Basin Trajectory Guard Summary"]
        lines.append(f"Prompt risk score: {self._risk_at_prompt:.3f}")
        lines.append(f"Tokens monitored: {len(self.diagnostics)}")
        if self.diagnostics:
            patched = sum(1 for d in self.diagnostics if d.was_patched)
            lines.append(f"Tokens patched: {patched}")
            avg_risk = sum(d.risk_score for d in self.diagnostics) / len(self.diagnostics)
            lines.append(f"Avg risk score: {avg_risk:.3f}")
            max_risk = max(self.diagnostics, key=lambda d: d.risk_score)
            lines.append(f"Max risk: {max_risk.risk_score:.3f} at token {max_risk.token_str!r}")
        return "\n".join(lines)


def demo():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing dependency: {e}\nInstall: pip install torch transformers")
        return

    print("Loading gpt2 for demonstration...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Factual trajectory examples
    factual = [
        "Gravity is a force of attraction between two objects with mass.",
        "Newton's law states every particle attracts every other particle.",
        "The strength of gravity depends on mass and distance.",
    ]

    # Known hallucination patterns (off-topic / confabulated)
    hallucinated = [
        "Otters hold hands while sleeping to avoid drifting apart.",
        "The moon is made of green cheese and NASA hid this fact.",
        "Einstein discovered gravity when an apple fell on his head in 1492.",
    ]

    guard = AttractorBasinGuard(
        model=model,
        tokenizer=tokenizer,
        factual_trajectory=factual,
        hallucinated_examples=hallucinated,
        config=BasinGuardConfig(
            target_layer=-2,
            regime_probe_layer=10,  # Adjust based on model depth
            patch_window_size=4,
            risk_threshold=0.5,
            lambda_max=1.0,
        ),
    )

    # Test prompts with varying risk profiles
    prompts = [
        "Explain why objects fall to the ground",
        "Tell me a fun fact about animals",
    ]

    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        output = guard.generate(prompt, max_new_tokens=25, do_sample=False)
        print(f"Output: {output}")
        print(guard.summary())

    guard.remove_hook()


if __name__ == "__main__":
    demo()
