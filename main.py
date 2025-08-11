import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import os
from sklearn import train_test_split


def load_and_preprocess_image(image_path, augment=False):
    image = tf.io.read_file(image_path)
    # Detect file type and decode accordingly
    # we are chosing between diffrent codec, png or jpeg
    if tf.strings.regex_full_match(image_path, ".*\\.png"):
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)
    
    image = tf.image.resize(image, [218, 178])  # Resize to 218 height x 178 width
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    # normalization for less data size
    
    return image

def load_data():
    # main dataset loading
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
    # Model with 3 convolutional layers 
    l2 = keras.regularizers.l2(0.001)
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(218, 178, 3)),
        keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l2),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=l2),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=l2),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(4, activation='sigmoid', kernel_regularizer=l2)  # 4 coordinates: left_x, left_y, right_x, right_y
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    # for rtx purposes
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    image_paths, labels = load_data()
    
    print(f"Found {len(image_paths)} images")
    labels = np.array(labels, dtype=np.float32)
    
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state=42)
    
    train_image_paths = [image_paths[i] for i in train_idx]
    val_image_paths = [image_paths[i] for i in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    print(f"Training set: {len(train_image_paths)} samples")
    print(f"Validation set: {len(val_image_paths)} samples")    # Funkcja do ładowania obrazów w locie
    def preprocess_train(path, label):
        image = load_and_preprocess_image(path)
        return image, label[:4]
    
    def preprocess_val(path, label):
        image = load_and_preprocess_image(path)
        return image, label[:4]

    model = create_model()
    print(model.summary())

    # loading trained model on RTX
    model_path = "model/eye_tracking_model_both_eyes.keras"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = keras.models.load_model(model_path, compile=False)
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

        # Create TensorFlow datasets for training and validation
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
        train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=100).batch(8).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
        val_dataset = val_dataset.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),  # Szybsze zatrzymanie
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, monitor='val_loss', min_lr=1e-7),
            keras.callbacks.ModelCheckpoint('model/best_both_eyes_model.keras', save_best_only=True, monitor='val_loss')
        ]

        print("Training both eyes model...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            callbacks=callbacks,
            verbose=1
        )

        model.save(model_path)
        print(f"Model saved to {model_path}")


    # Example prediction on validation set
    if len(val_image_paths) > 0:
        example_path = "photos/matt/3.jpg"
        example_image = load_and_preprocess_image(example_path)
        
        # Get true labels for this specific image from validation set
        example_label_both_eyes = labels[154][:4]  # Only left and right eye coordinates

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

        # Poprawka: zamiana list na tablice numpy przed odejmowaniem
        left_true = np.array([example_label_both_eyes[0], example_label_both_eyes[1]])
        left_pred = np.array([predicted_landmarks[0][0], predicted_landmarks[0][1]])
        right_true = np.array([example_label_both_eyes[2], example_label_both_eyes[3]])
        right_pred = np.array([predicted_landmarks[0][2], predicted_landmarks[0][3]])
        errorLeft = np.mean(np.abs(left_true - left_pred)) * 178
        errorRight = np.mean(np.abs(right_true - right_pred)) * 178
        print(f"Mean absolute error (pixels) Left eye: {errorLeft:.2f}")
        print(f"Mean absolute error (pixels) Right eye: {errorRight:.2f}")

if __name__ == "__main__":
    main()
