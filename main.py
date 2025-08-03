import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_and_preprocess_image(image_path, augment=False):
    """Load and preprocess a single image"""
    image = tf.io.read_file(image_path)
    # Detect file type and decode accordingly
    if tf.strings.regex_full_match(image_path, ".*\\.png"):
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)
    
    image = tf.image.resize(image, [218, 178])  # Resize to 218 height x 178 width
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    
    # More aggressive data augmentation for training with larger dataset
    if augment:
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        # Random saturation and hue
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_hue(image, 0.1)
        # Random jpeg quality (simulation)
        if tf.random.uniform([]) < 0.3:
            image = tf.image.random_jpeg_quality(image, 70, 100)
        # Ensure values stay in [0,1]
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

def augment_data(images, labels, augmentation_factor=2):
    """Create augmented dataset by applying random transformations"""
    augmented_images = []
    augmented_labels = []
    
    for i in range(len(images)):
        # Add original image
        augmented_images.append(images[i])
        augmented_labels.append(labels[i])
        
        # Add augmented versions
        for _ in range(augmentation_factor):
            # Apply augmentation to image
            aug_image = tf.image.random_brightness(images[i], 0.2)
            aug_image = tf.image.random_contrast(aug_image, 0.7, 1.3)
            aug_image = tf.image.random_saturation(aug_image, 0.7, 1.3)
            aug_image = tf.image.random_hue(aug_image, 0.1)
            aug_image = tf.clip_by_value(aug_image, 0.0, 1.0)
            
            augmented_images.append(aug_image)
            augmented_labels.append(labels[i])  # Labels stay the same
    
    return tf.stack(augmented_images), np.array(augmented_labels)

def load_data():
    df = pd.read_csv("dataset/list_landmarks_align_celeba.txt", sep=" ", skipinitialspace=True)
    image_paths = ["photos/hongkong/" + name for name in df["filename"]]
    
    # Verify all image files exist
    missing_files = [path for path in image_paths if not os.path.exists(path)]
    if missing_files:
        print(f"Warning: Missing {len(missing_files)} files out of {len(image_paths)} total")
        print(f"First 5 missing files: {missing_files[:5]}")
    
    # Filter out missing files
    valid_indices = [i for i, path in enumerate(image_paths) if os.path.exists(path)]
    image_paths = [image_paths[i] for i in valid_indices]
    
    print(f"Found {len(image_paths)} valid images out of {len(df)} entries in CSV")
    
    # Get corresponding labels and normalize coordinates
    labels = df[['lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']].iloc[valid_indices].values.astype("float32")
    # Normalize coordinates to [0, 1] range
    labels[:, [0, 2, 4, 6, 8]] = labels[:, [0, 2, 4, 6, 8]] / 178.0  # x coordinates (width)
    labels[:, [1, 3, 5, 7, 9]] = labels[:, [1, 3, 5, 7, 9]] / 218.0  # y coordinates (height)

    return image_paths, labels

def create_model():
    # More sophisticated model for larger dataset (3000 images)
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(218, 178, 3)),
        
        # First conv block
        keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Third conv block
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Fourth conv block
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(4, activation='sigmoid')  # 4 coordinates: left_x, left_y, right_x, right_y
    ])
    
    # Higher learning rate for larger dataset
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, 
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    image_paths, labels = load_data()
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    images = tf.stack([load_and_preprocess_image(path) for path in image_paths])
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Proper train/validation split for larger dataset
    from sklearn.model_selection import train_test_split
    
    # Split data: 80% train, 20% validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        images.numpy(), labels, test_size=0.2, random_state=42
    )
    
    # Apply augmentation only to training data
    print("Applying data augmentation to training set...")
    train_images_aug, train_labels_aug = augment_data(train_images, train_labels, augmentation_factor=1)
    
    print(f"Training set: {train_images_aug.shape[0]} samples (with augmentation)")
    print(f"Validation set: {val_images.shape[0]} samples (original data)")
    
    # Create single model for both eyes
    model = create_model()
    print(model.summary())

    # Check if model exists
    model_path = "model/eye_tracking_model_both_eyes.keras"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = keras.models.load_model(model_path, compile=False)
        
        # Recompile the model with the same settings
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer, 
            loss='mse',
            metrics=['mae']
        )
    else:
        print("Training new model...")
        if not os.path.exists("model"):
            os.makedirs("model")
    
        
        # Create dataset with both eyes coordinates (left_x, left_y, right_x, right_y)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images_aug.numpy(), train_labels_aug[:, :4]))  # Both eyes x,y
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels[:, :4]))  # Both eyes x,y
        val_dataset = val_dataset.batch(32)

        # Callbacks for better training with larger dataset
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, monitor='val_loss', min_lr=1e-7),
            keras.callbacks.ModelCheckpoint('model/best_both_eyes_model.keras', save_best_only=True, monitor='val_loss')
        ]
        
        print("Training both eyes model...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,  # Reduced epochs since we have more data
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model using SavedModel format (more reliable)
        model.save(model_path)
        print(f"Model saved to {model_path}")


    # Example prediction on validation set
    if len(val_images) > 0:
        example_path = "photos/matt/1.jpg"  # Use a valid image path from the dataset
        example_image = load_and_preprocess_image(example_path)
        
        # Get true labels for this specific image from validation set
        example_label_both_eyes = [0, 0, 0, 0]  # Placeholder: [left_x, left_y, right_x, right_y]

        # Single prediction for both eyes
        predicted_landmarks = model.predict(tf.expand_dims(example_image, axis=0))

        print("\nPrediction example:")
        print("True landmarks (Left) x:", example_label_both_eyes[0] * 178, " True landmarks (Left) y:", example_label_both_eyes[1] * 218)  # Denormalize for display
        print("Predicted landmarks (Left) x:", predicted_landmarks[0][0] * 178, "Predicted landmarks (Left) y:", predicted_landmarks[0][1] * 218)  # Denormalize for display
        print("True landmarks (Right) x:", example_label_both_eyes[2] * 178, " True landmarks (Right) y:", example_label_both_eyes[3] * 218)  # Denormalize for display
        print("Predicted landmarks (Right) x:", predicted_landmarks[0][2] * 178, "Predicted landmarks (Right) y:", predicted_landmarks[0][3] * 218)  # Denormalize for display

        # Plot rectangles (boxes) around eyes
        plt.figure(figsize=(12, 10))
        plt.imshow(example_image.numpy())
        plt.title("Eye Detection Results with Bounding Boxes", fontsize=16, fontweight='bold')
        
        # Get current axes for adding rectangles
        ax = plt.gca()
        
        # Define box size (in pixels)
        box_width = 25
        box_height = 20
        
        # Calculate box coordinates for predicted eyes (center the box on the predicted point)
        pred_left_x = predicted_landmarks[0][0] * 178
        pred_left_y = predicted_landmarks[0][1] * 218
        pred_right_x = predicted_landmarks[0][2] * 178
        pred_right_y = predicted_landmarks[0][3] * 218
        
        # Calculate box coordinates for true eyes
        true_left_x = example_label_both_eyes[0] * 178
        true_left_y = example_label_both_eyes[1] * 218
        true_right_x = example_label_both_eyes[2] * 178
        true_right_y = example_label_both_eyes[3] * 218
        
        # Add rectangles for predicted eyes
        from matplotlib.patches import Rectangle
        
        # Predicted left eye box (red)
        pred_left_box = Rectangle((pred_left_x - box_width/2, pred_left_y - box_height/2), 
                                 box_width, box_height, 
                                 linewidth=3, edgecolor='red', facecolor='none', 
                                 label='Predicted Left Eye')
        ax.add_patch(pred_left_box)
        
        # Predicted right eye box (blue)
        pred_right_box = Rectangle((pred_right_x - box_width/2, pred_right_y - box_height/2), 
                                  box_width, box_height, 
                                  linewidth=3, edgecolor='blue', facecolor='none', 
                                  label='Predicted Right Eye')
        ax.add_patch(pred_right_box)
        
        # Add rectangles for true eyes (with dashed style)
        # True left eye box (green, dashed)
        true_left_box = Rectangle((true_left_x - box_width/2, true_left_y - box_height/2), 
                                 box_width, box_height, 
                                 linewidth=2, edgecolor='green', facecolor='none', 
                                 linestyle='--', label='True Left Eye')
        ax.add_patch(true_left_box)
        
        # True right eye box (yellow, dashed)
        true_right_box = Rectangle((true_right_x - box_width/2, true_right_y - box_height/2), 
                                  box_width, box_height, 
                                  linewidth=2, edgecolor='yellow', facecolor='none', 
                                  linestyle='--', label='True Right Eye')
        ax.add_patch(true_right_box)
        
        # Add center points for better visibility
        plt.plot(pred_left_x, pred_left_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1)
        plt.plot(pred_right_x, pred_right_y, 'bo', markersize=8, markeredgecolor='white', markeredgewidth=1)
        plt.plot(true_left_x, true_left_y, 'g^', markersize=8, markeredgecolor='white', markeredgewidth=1)
        plt.plot(true_right_x, true_right_y, 'y^', markersize=8, markeredgecolor='white', markeredgewidth=1)
        
        # Add crosshairs for better visibility (optional - you can remove these if boxes are enough)
        plt.axhline(y=pred_left_y, color='red', linestyle=':', alpha=0.2)
        plt.axvline(x=pred_left_x, color='red', linestyle=':', alpha=0.2)
        plt.axhline(y=pred_right_y, color='blue', linestyle=':', alpha=0.2)
        plt.axvline(x=pred_right_x, color='blue', linestyle=':', alpha=0.2)
        
        plt.legend(loc='upper right', fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Calculate error for both eyes
        errorLeft = np.mean(np.abs([example_label_both_eyes[0], example_label_both_eyes[1]] - [predicted_landmarks[0][0], predicted_landmarks[0][1]])) * 178
        errorRight = np.mean(np.abs([example_label_both_eyes[2], example_label_both_eyes[3]] - [predicted_landmarks[0][2], predicted_landmarks[0][3]])) * 178
        print(f"Mean absolute error (pixels) Left eye: {errorLeft:.2f}")
        print(f"Mean absolute error (pixels) Right eye: {errorRight:.2f}")

if __name__ == "__main__":
    main()
