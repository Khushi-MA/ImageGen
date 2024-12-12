import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A

class HandSignPreprocessor:
    def __init__(self, root_dir="sample", test_size=0.2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.root_dir = Path(root_dir)
        self.test_size = test_size
        
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.3),
            A.RandomRotate90(p=0.2),
            A.Flip(p=0.2)
        ])

    def _extract_landmarks(self, image):
        """Extract normalized hand landmarks from image"""
        # Resize image to consistent dimensions
        TARGET_WIDTH = 640
        TARGET_HEIGHT = 480
        
        # Maintain aspect ratio while resizing
        h, w = image.shape[:2]
        scale = min(TARGET_WIDTH/w, TARGET_HEIGHT/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Add padding if needed to reach target size
        delta_w = TARGET_WIDTH - new_w
        delta_h = TARGET_HEIGHT - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=[0,0,0])
        
        # Extract landmarks from padded image
        results = self.hands.process(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        
        # Get landmarks and normalize
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
        
        # Normalize coordinates relative to image dimensions
        landmarks[:, 0] = landmarks[:, 0] * TARGET_WIDTH  # x coords
        landmarks[:, 1] = landmarks[:, 1] * TARGET_HEIGHT # y coords
        
        # Center and scale
        landmarks = landmarks - landmarks.mean(axis=0)
        landmarks = landmarks / np.abs(landmarks).max()
        
        return landmarks.flatten()

    def process_dataset(self):
        data, labels = [], []
        
        for user_dir in self.root_dir.glob("u*"):
            for img_path in user_dir.glob("*.jpg"):
                letter = img_path.stem[0]
                image = cv2.imread(str(img_path))
                
                if image is None:
                    continue
                    
                augmented = self.transform(image=image)["image"]
                landmarks = self._extract_landmarks(augmented)
                
                if landmarks is not None:
                    data.append(landmarks)
                    labels.append(letter)

        X = np.array(data)
        y = np.array(labels)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=42
        )
        
        return {
            'X_train': torch.FloatTensor(X_train),
            'X_val': torch.FloatTensor(X_val),
            'y_train': y_train,
            'y_val': y_val,
            'letter_to_idx': {l: i for i, l in enumerate(np.unique(y))}
        }

class HandSignDataset(Dataset):
    def __init__(self, X, y, letter_to_idx):
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got X: {len(X)}, y: {len(y)}")
        if not all(label in letter_to_idx for label in y):
            invalid_labels = [l for l in y if l not in letter_to_idx]
            raise ValueError(f"Invalid labels found: {invalid_labels}")
            
        self.X = X
        self.y = torch.LongTensor([letter_to_idx[label] for label in y])
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return self.X[idx], self.y[idx]


def train_model(data, epochs=200, batch_size=16, lr=0.001):
    try:
        train_dataset = HandSignDataset(data['X_train'], data['y_train'], data['letter_to_idx'])
        val_dataset = HandSignDataset(data['X_val'], data['y_val'], data['letter_to_idx'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = HandSignModel(
            input_size=63,  # 21 landmarks * 3 coordinates
            num_classes=len(data['letter_to_idx'])
        ).to(device)  # Move to GPU if available
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()
            
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            accuracy = 100. * correct / total
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {accuracy:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'letter_to_idx': data['letter_to_idx'],
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'accuracy': accuracy
                }
                torch.save(best_model_state, 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
                
            scheduler.step(val_loss)
        
        return model  # Ensure model is returned
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None


def generate_hand_landmarks(letter, model_path='best_model.pth'):
    # Load checkpoint with safety flag
    checkpoint = torch.load(
        model_path, 
        map_location=torch.device('cpu'),
        weights_only=True
    )
    letter_to_idx = checkpoint['letter_to_idx']
    
    if letter not in letter_to_idx:
        raise ValueError(f"Letter '{letter}' not supported. Available letters: {list(letter_to_idx.keys())}")
    
    # Create model with same architecture as training
    model = HandSignModel(
        input_size=63,  # 21 landmarks * 3 coordinates
        hidden_size=512,
        num_classes=len(letter_to_idx)  # Number of letters
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        # Create input tensor from landmarks
        input_tensor = torch.zeros(1, 63)  # Batch size 1, 63 features
        # Generate output
        output = model(input_tensor)
        # Convert to landmarks shape
        landmarks = output.numpy().reshape(21, 3)  # 21 landmarks, 3 coordinates each
    
    return landmarks

class HandSignModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=512, num_classes=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size//2, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A

# Define device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Initializing preprocessor...")
        preprocessor = HandSignPreprocessor(root_dir="sample")
        
        logger.info("Processing dataset...")
        data = preprocessor.process_dataset()
        logger.info(f"Training samples: {len(data['X_train'])}")
        logger.info(f"Validation samples: {len(data['X_val'])}")
        logger.info(f"Unique letters: {list(data['letter_to_idx'].keys())}")

        logger.info("Training model...")
        model = train_model(data, epochs=200, batch_size=16, lr=0.001)
        
        if model is None:
            raise RuntimeError("Model training failed")

        logger.info("Saving model...")
        save_path = Path("models")
        save_path.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'letter_to_idx': data['letter_to_idx']
        }, save_path / "final_model.pth")

        logger.info("Testing generation...")
        for letter in data['letter_to_idx'].keys():
            landmarks = generate_hand_landmarks(letter, model_path=str(save_path / "final_model.pth"))
            if landmarks is not None:
                logger.info(f"Successfully generated landmarks for letter '{letter}'")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()