from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.models import ResNet50_Weights


class AdversarialAttack:
    def __init__(self, model_name="resnet50"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained model from torchvision
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=self.weights)
        self.model = self.model.to(self.device)
        self.model.eval()  # eval mode

        # The standard preprocessing steps for ResNet50 require specific image transformations:
        # Input Requirements
        # Images must be 3-channel RGB
        # Minimum resolution of 224x224
        # Pixel values must be in range [0, 1]
        # self.transform = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )
        # TorchVision bundles the necessary preprocessing transforms into each model weight
        self.transform = self.weights.transforms()

        # Extract normalization parameters from the transform pipeline
        self.normalize_mean = (
            torch.tensor(self.transform.mean).view(3, 1, 1).to(self.device)
        )
        self.normalize_std = (
            torch.tensor(self.transform.std).view(3, 1, 1).to(self.device)
        )

        # Load ImageNet class labels
        self.classes = self.weights.meta["categories"]

        # Load ImageNet class labels from file  (practically the same)
        # with open("imagenet_classes.txt") as f:
        #     self.classes = [line.strip() for line in f.readlines()]

    def normalize_image(self, image_tensor):
        """Convert image from [0,1] range to normalized range."""
        return (image_tensor - self.normalize_mean) / self.normalize_std

    def denormalize_image(self, image_tensor):
        """Convert image from normalized range back to [0,1] range."""
        return image_tensor * self.normalize_std + self.normalize_mean

    def load_image(self, image_path):
        """Load and preprocess image."""

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)

    def save_image(self, tensor, path):
        """Save tensor as image."""

        # Denormalize tensor
        tensor = self.denormalize_image(tensor)

        # Convert to PIL Image
        tensor = tensor.cpu().squeeze(0)
        tensor = tensor.clip(0, 1)
        tensor = transforms.ToPILImage()(tensor)
        tensor.save(path)

    def get_prediction(self, image_tensor):
        """Get model prediction for image."""

        with torch.no_grad():
            output = self.model(image_tensor)

        _, predicted_idx = torch.max(output, 1)
        predicted_class = self.classes[predicted_idx.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx].item()

        return predicted_class, confidence

    def generate_adversarial_example(
        self, image_tensor, target_class_idx, epsilon, max_iterations
    ):
        """Generate adversarial example using iterative FGSM.

        Args:
            image_tensor: Input image tensor
            target_class_idx: Target class index
            epsilon: Step size for perturbation
            max_iterations: Maximum number of iterations

        Returns:
            Adversarial example tensor, number of iterations taken
        """

        # Convert epsilon to normalized space
        epsilon_normalized = epsilon / self.normalize_std

        # Clone the image tensor to avoid modifying the original
        perturbed_image = image_tensor.clone().detach().requires_grad_(True)

        # Use negative loss to maximize the target class probability, so we use negative loss
        criterion = nn.CrossEntropyLoss()
        target = torch.tensor([target_class_idx]).to(self.device)

        best_confidence = float("-inf")
        best_image = None

        for i in range(max_iterations):
            perturbed_image.requires_grad_(True)

            # Forward pass
            output = self.model(perturbed_image)

            # Calculate loss to maximize target class probability
            loss = -criterion(output, target)  # Negative loss for maximization

            # Get current predictions and confidence
            with torch.no_grad():
                probs = torch.nn.functional.softmax(output, dim=1)
                current_confidence = probs[0][target_class_idx].item()
                _, predicted = torch.max(output, 1)

                # Store best result
                if current_confidence > best_confidence:
                    best_confidence = current_confidence
                    best_image = perturbed_image.clone().detach()

                # Print progress
                if i % 10 == 0:
                    print(
                        f"Iteration {i}: Target class probability: {current_confidence:.4f}"
                    )

                # Return if target class is achieved early
                if predicted.item() == target_class_idx:
                    print(f"Success! Target class achieved at iteration {i}")
                    return perturbed_image.detach(), i + 1

            # Backward pass
            loss.backward()

            # Compute perturbation in normalized space
            data_grad = perturbed_image.grad.data
            perturbation = epsilon_normalized * torch.sign(data_grad)

            # Add perturbation to image
            perturbed_image = perturbed_image.detach() + perturbation

            # Clip to maintain valid pixel range in normalized space
            zero_normed = [
                -m / s for m, s in zip(self.normalize_mean, self.normalize_std)
            ]
            max_normed = [
                (1 - m) / s for m, s in zip(self.normalize_mean, self.normalize_std)
            ]
            zero_normed = (
                torch.tensor(zero_normed, dtype=torch.float)
                .view(3, 1, 1)
                .to(self.device)
            )
            max_normed = (
                torch.tensor(max_normed, dtype=torch.float)
                .view(3, 1, 1)
                .to(self.device)
            )
            perturbed_image = torch.clamp(
                perturbed_image, min=zero_normed, max=max_normed
            )

            # Reset gradients
            if perturbed_image.grad is not None:
                perturbed_image.grad.zero_()

        print(
            f"\nAttack did not achieve target class. Best confidence: {best_confidence:.4f}"
        )
        return (
            best_image if best_image is not None else perturbed_image.detach(),
            max_iterations,
        )


    def save_perturbation_visualization(
        self, original, perturbed, path, amplification=10, center=0.5
    ):
        """Save an enhanced visualization of the perturbation.

        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            path: Path to save the visualization
            amplification: Factor to amplify the perturbation (default: 10)
            center: Value to center the perturbation around (default: 0.5)
        """
        # Calculate raw perturbation
        perturbation = perturbed - original

        # Compute perturbation statistics for normalization
        max_abs_val = torch.max(torch.abs(perturbation))

        # Normalize perturbation to [-1, 1] range
        normalized_perturbation = perturbation / (max_abs_val + 1e-8)

        # Create heatmap tensor with same shape as input
        heatmap = torch.zeros_like(perturbed)

        # Create masks for positive and negative perturbations
        pos_mask = normalized_perturbation > 0
        neg_mask = normalized_perturbation < 0

        # Apply amplification and center around 0.5
        enhanced_perturbation = normalized_perturbation * amplification + center
        # enhanced_perturbation = torch.clamp(enhanced_perturbation, 0, 1)

        # Red channel: Emphasize negative perturbations
        heatmap[:, 0] = torch.where(
            neg_mask[:, 0], 1 - enhanced_perturbation[:, 0], enhanced_perturbation[:, 0]
        )

        # Green channel: Emphasize positive perturbations
        heatmap[:, 1] = torch.where(
            pos_mask[:, 0], enhanced_perturbation[:, 0], 1 - enhanced_perturbation[:, 0]
        )

        # Blue channel: Reduced to emphasize red/green contrast
        heatmap[:, 2] = 0.5 * enhanced_perturbation[:, 0]

        # Save the enhanced visualization
        self.save_image(heatmap, path)

        # Save perturbation statistics
        stats = {
            "max_absolute_perturbation": max_abs_val.item(),
            "mean_absolute_perturbation": torch.mean(torch.abs(perturbation)).item(),
            "std_perturbation": torch.std(perturbation).item(),
        }

        return stats


def main():
    # Set up argument parser

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate adversarial examples using FGSM"
    )
    parser.add_argument(
        "--image", "-i", type=Path, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--target-class",
        "-t",
        type=int,
        required=True,
        help="""Target class index (0-999 for ImageNet),
            check imagenet_classes.txt for reference (mind the offset!)
        """,
    )
    parser.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=0.0001,
        help="""Epsilon value for FGSM (default: 0.01)
            This determines the magnitude of the perturbation
            Larger values result in more noticeable changes
            Smaller values are less noticeable but may require more iterations
        """,
    )
    parser.add_argument(
        "--max-iterations",
        "-m",
        type=int,
        default=300,
        help="Maximum number of iterations (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default="output",
        help="Output directory for saving results (default: output)",
    )

    def get_class_from_file(index, classes_file="imagenet_classes.txt"):
        try:
            with open(classes_file, "r") as file:
                lines = file.readlines()
                if 0 <= index < len(lines):
                    return lines[index].strip()
                else:
                    print("Index out of file range")
        except FileNotFoundError:
            print(f"Classes file {classes_file} not found.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize attack
    attack = AdversarialAttack()

    # Check if image exists
    if not args.image.is_file():
        print(f"Error: Image file '{args.image}' not found")
        return

    try:
        image_tensor = attack.load_image(str(args.image))
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Get original prediction
    original_class, original_conf = attack.get_prediction(image_tensor)
    print(f"\nOriginal prediction: {original_class} ({original_conf:.2%})")

    # Generate adversarial example
    print(f"\nGenerating adversarial example...")
    print(f"Target class index: {args.target_class}")
    print(f"Target class name: {get_class_from_file(args.target_class)}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Max iterations: {args.max_iterations}")

    perturbed_image, iterations = attack.generate_adversarial_example(
        image_tensor,
        args.target_class,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
    )

    # Get new prediction
    adversarial_class, adv_conf = attack.get_prediction(perturbed_image)
    print(f"\nAdversarial prediction: {adversarial_class} ({adv_conf:.2%})")
    print(f"Number of iterations: {iterations}")

    # Save results
    original_path = args.output_dir / "original.png"
    adversarial_path = args.output_dir / "adversarial.png"
    perturbation_path = args.output_dir / "perturbation.png"

    print(f"\nSaving results to {args.output_dir}/")
    attack.save_image(image_tensor, original_path)
    attack.save_image(perturbed_image, adversarial_path)

    # Save perturbation
    attack.save_perturbation_visualization(
        image_tensor, perturbed_image, perturbation_path, amplification=20, center=0.5
    )

    print("Output files:")
    print(f"- Original image: {original_path}")
    print(f"- Adversarial image: {adversarial_path}")
    print(f"- Perturbation visualization: {perturbation_path}")

    # TODO Improve the perturbation to make it more subtle


if __name__ == "__main__":
    main()
