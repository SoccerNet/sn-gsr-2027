"""Jersey number and color reading with the Qwen3-VL vision-language model."""

import re

import torch
from PIL import Image

# Same color vocabulary as the CLIP color head (see sngsr.tracking.JERSEY_COLORS).
JERSEY_COLORS = [
    "Red", "Blue", "Green", "Yellow", "White", "Black", "Orange",
    "Purple", "Pink", "Brown", "Grey", "Charcoal", "Neon yellow",
]
NOT_VISIBLE = 100  # sentinel number: the jersey number cannot be read reliably

PROMPT = (
    "Look at this football player's jersey.\n"
    "\n"
    "STRICT RULES — follow them exactly:\n"
    "- The player MUST be seen either directly from behind (full back visible) "
    "OR directly from the front (front torso/thigh visible). "
    "If the player is seen from the side or at a 3/4 angle (digits could be "
    "distorted, e.g. an 8 looking like a 3, or an 84 looking like a 34), "
    "return: 100,Unknown\n"
    "- Every digit of the number MUST be fully visible, sharp, and unambiguous. "
    "If any digit is cropped, folded, hidden by an arm/another player/the image "
    "border, or ambiguous, return: 100,Unknown\n"
    "- Do NOT guess. When in doubt, return 100,Unknown — it is better to refuse "
    "than to predict a wrong number.\n"
    "\n"
    "Otherwise, answer with EXACTLY two values separated by a comma and nothing else:\n"
    "1) the jersey number as an integer\n"
    "2) the jersey color, chosen STRICTLY from this list: "
    f"{', '.join(JERSEY_COLORS)}\n"
    "Format: <number>,<color>\n"
    "Example: 10,Red"
)


def normalize_color(raw: str) -> str:
    """Map a free-form answer onto JERSEY_COLORS (case-insensitive), else 'Unknown'."""
    lower = raw.strip().strip(".").strip().lower()
    for c in JERSEY_COLORS:
        if c.lower() == lower:
            return c
    for c in JERSEY_COLORS:
        if c.lower() in lower:
            return c
    return "Unknown"


def parse_answer(raw: str) -> tuple[int, str]:
    """Parse a '<number>,<color>' answer; anything else is (NOT_VISIBLE, 'Unknown')."""
    parts = (raw.splitlines()[0] if raw else "").split(",")
    if len(parts) < 2:
        return NOT_VISIBLE, "Unknown"
    m = re.search(r"-?\d+", parts[0])
    number = int(m.group(0)) if m else NOT_VISIBLE
    return number, normalize_color(parts[1])


def fp8_supported(device) -> bool:
    """FP8 kernels need CUDA compute capability >= 8.9 (Ada, Hopper, Blackwell)."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) >= (8, 9)


def resolve_model_id(model_id: str, device) -> str:
    """Use the BF16 checkpoint when an FP8 one is requested on hardware without FP8.

    Same weights, same answers up to quantization; only needs more memory
    (~17 GB instead of ~10 GB for the 8B model).
    """
    if model_id.upper().endswith("-FP8") and not fp8_supported(device):
        fallback = model_id[: -len("-FP8")]
        print(f"WARNING: {model_id} needs a GPU with compute capability >= 8.9; using {fallback} instead. "
              "Set QWEN.MODEL_ID to a smaller checkpoint (e.g. Qwen/Qwen3-VL-2B-Instruct) "
              "if it does not fit in memory.", flush=True)
        return fallback
    return model_id


class QwenJerseyReader:
    """Batched jersey (number, color) reading from player crops."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct-FP8", max_new_tokens: int = 24,
                 device="cuda"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        device = torch.device(device)
        model_id = resolve_model_id(model_id, device)
        # BF16 needs Ampere or newer; older GPUs get FP16, the CPU FP32.
        if device.type == "cuda":
            dtype = "auto" if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        print(f"Loading {model_id} on {device} ...", flush=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map={"": str(device)}, attn_implementation="eager",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Left padding is required for batched generation (generated tokens all
        # start right after input_ids.shape[1]).
        self.processor.tokenizer.padding_side = "left"
        self.max_new_tokens = max_new_tokens
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        self.prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def read(self, images: list[Image.Image]) -> list[tuple[int, str]]:
        """N crops -> N (number, color) in a single forward pass."""
        if not images:
            return []
        inputs = self.processor(
            text=[self.prompt] * len(images), images=list(images),
            return_tensors="pt", padding=True,
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        answers = self.processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return [parse_answer(a.strip()) for a in answers]
