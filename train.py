import torch

import torch.nn as nn


from dataset import (
    train_loader,
    val_loader
)


from model import (
    VGG,
    get_vgg_layers,
    vgg11_config
)


from config import (
    EPOCHS,
    LEARNING_RATE
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)

print(
    "Device:",
    device
)


# ==========================================
# CREATE MODEL
# ==========================================

output_dim = 2


vgg_layers = get_vgg_layers(

    vgg11_config,

    batch_norm=True
)


model = VGG(

    vgg_layers,

    output_dim
)


model = model.to(device)


print(model)


# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

best_val_accuracy = 0.0


# ==========================================
# STORE HISTORY
# ==========================================

train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []


# ==========================================
# TRAINING
# ==========================================

for epoch in range(EPOCHS):


    # ======================================
    # TRAIN MODE
    # ======================================

    model.train()


    running_train_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:


        # Move data to GPU/CPU
        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)


        # Calculate loss
        loss = criterion(

            outputs,

            labels

        )


        # Remove old gradients
        optimizer.zero_grad()


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Add loss
        running_train_loss += loss.item()


        # Prediction
        _, predicted = torch.max(

            outputs,

            1

        )


        train_total += labels.size(0)


        train_correct += (

            predicted == labels

        ).sum().item()


    # ======================================
    # TRAIN METRICS
    # ======================================

    train_loss = (

        running_train_loss

        / len(train_loader)

    )


    train_accuracy = (

        100
        * train_correct
        / train_total

    )


    # ======================================
    # VALIDATION MODE
    # ======================================

    model.eval()


    running_val_loss = 0.0

    val_correct = 0

    val_total = 0


    # No gradient during validation
    with torch.no_grad():


        for images, labels in val_loader:


            images = images.to(device)

            labels = labels.to(device)


            # Forward pass
            outputs = model(images)


            # Validation loss
            loss = criterion(

                outputs,

                labels

            )


            running_val_loss += loss.item()


            # Prediction
            _, predicted = torch.max(

                outputs,

                1

            )


            val_total += labels.size(0)


            val_correct += (

                predicted == labels

            ).sum().item()


    # ======================================
    # VALIDATION METRICS
    # ======================================

    val_loss = (

        running_val_loss

        / len(val_loader)

    )


    val_accuracy = (

        100
        * val_correct
        / val_total

    )


    # ======================================
    # SAVE HISTORY
    # ======================================

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # ======================================
    # PRINT EVERYTHING TOGETHER
    # ======================================

    print(

        f"Epoch [{epoch + 1}/{EPOCHS}] | "

        f"Train Loss: {train_loss:.4f} | "

        f"Val Loss: {val_loss:.4f} | "

        f"Train Acc: {train_accuracy:.2f}% | "

        f"Val Acc: {val_accuracy:.2f}%"

    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_val_accuracy:


        best_val_accuracy = val_accuracy


        torch.save(

            {

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "class_names":
                    [
                        "cloud",
                        "not_cloud"
                    ]

            },

            "best_vgg11_cloud.pth"

        )


        print(
            "Best model saved!"
        )


# ==========================================
# TRAINING COMPLETE
# ==========================================

print()

print(
    "Training completed!"
)

print(

    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"

)