import torch

from PIL import Image

from torchvision import transforms


from model import (
    VGG,
    get_vgg_layers,
    vgg11_config
)


device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)

model = VGG(

    get_vgg_layers(
        vgg11_config,
        batch_norm=True
    ),

    output_dim=2

)



checkpoint = torch.load(

    "best_vgg11_cloud.pth",

    map_location=device

)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]

)


model = model.to(device)

model.eval()


transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_image(image_path):


    # Open image
    image = Image.open(
        image_path
    )


    # Convert to RGB
    image = image.convert(
        "RGB"
    )


    # Transform
    image = transform(
        image
    )


    # Add batch dimension
    image = image.unsqueeze(
        0
    )


    # GPU
    image = image.to(
        device
    )


    # Prediction
    with torch.no_grad():


        outputs = model(
            image
        )


        # Convert logits to probabilities
        probabilities = torch.softmax(

            outputs,

            dim=1

        )


        probability, predicted = torch.max(

            probabilities,

            1

        )


    # Classes
    class_names = [
        "cloud",
        "not_cloud"
    ]


    predicted_class = class_names[

        predicted.item()

    ]


    confidence = (

        probability.item()
        * 100

    )


    print()

    print(
        "Prediction:",
        predicted_class
    )

    print(

        f"Confidence: "
        f"{confidence:.2f}%"

    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":


    image_path = input(

        "Enter image path: "

    )


    predict_image(
        image_path
    )