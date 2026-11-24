"""CLIP ViT-L/14 with role, jersey-number (two digits) and jersey-color heads."""
import os

import clip
import torch
import torch.nn as nn

class CLIPFinetune(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        num_classes_role: int = 5,
        num_classes_number: int = 11,  # 0-10s
        color_embedding_dim: int = 768,  # 512 for vit-B/32,
        freeze_clip: bool = True,
        device="cuda",
    ):
        super().__init__()
        # Load CLIP model
        # CLIP_HOME relocates the backbone cache (default ~/.cache/clip), e.g. in Docker.
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device, download_root=os.environ.get("CLIP_HOME"))
        self.clip_feature_dim = self.clip_model.visual.output_dim

        # Freeze CLIP parameters if specified
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Task-specific heads
        # 1. Role classifier (player, goalkeeper, referee, ball, other)
        self.role_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_role)
        )

        # 2. Jersey number classifiers: one for unit (ones) and the other for tens digits (each: 0-10)
        self.digit1_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_number)
        )
        self.digit2_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_number)
        )

        # 3. Jersey color encoder
        # We'll use CLIP's text encoder to create color embeddings
        self.color_projection = nn.Linear(self.clip_feature_dim, color_embedding_dim)

    def encode_image(self, image):
        with torch.no_grad() if self.clip_model.visual.parameters().__next__().requires_grad == False else torch.enable_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features

    def encode_color_text(self, color_text):
        with torch.no_grad():
            color_tokens = clip.tokenize(color_text).to(next(self.parameters()).device)
            color_features = self.clip_model.encode_text(color_tokens)
        return color_features

    def forward(self, images):
        # Get image features from CLIP
        image_features = self.encode_image(images)
        image_features = image_features.float()

        # Get predictions for each task
        role_logits = self.role_classifier(image_features)
        digit1_logits = self.digit1_classifier(image_features)
        digit2_logits = self.digit2_classifier(image_features)
        color_embedding = self.color_projection(image_features)

        return {
            'role_logits': role_logits,
            'digit1_logits': digit1_logits,
            'digit2_logits': digit2_logits,
            'color_embedding': color_embedding,
        }


