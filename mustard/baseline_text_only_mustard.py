import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import json
import os
from pathlib import Path

# CONFIGURATION
MUSTARD_DIR = r'mustard_dataset'
FEATURES_DIR = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED' 
SPLITS_FILE = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'

# Context and punchline paths
CONTEXT_PATHS = {
    'sarcasm': f'{FEATURES_DIR}/context/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/context/not_sarcasm'
}
PUNCHLINE_PATHS = {
    'sarcasm': f'{FEATURES_DIR}/utterances/sarcasm', 
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm'
}

# Text feature variants: context+punchline vs punchline-only
TEXT_FEATURES = [
    'bert_context_punchline',
    'e5_context_punchline', 
    'bert_punchline_only',
    'e5_punchline_only'
]

FEATURE_DIMS = {
    'bert_context_punchline': 1536,  # 768 + 768
    'e5_context_punchline': 2048,    # 1024 + 1024
    'bert_punchline_only': 768,      # 768 only
    'e5_punchline_only': 1024        # 1024 only
}

LABEL_COLUMN = 'label'
NUM_CLASSES = 2  # Binary: sarcasm vs not_sarcasm
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
SEED = 999

PREDICTIONS_DIR = 'predictions_output'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, x, y, file_names): 
        self.x, self.y, self.file_names = x, y, file_names
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, i): 
        return torch.tensor(self.x[i]), torch.tensor(self.y[i], dtype=torch.long), self.file_names[i]

class TextModel(nn.Module):
    def __init__(self, input_dim):
        super(TextModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
    def forward(self, x): 
        return self.net(x)

def load_mustard_splits():
    """Load MUStARD 5-fold cross-validation splits"""
    try:
        with open(SPLITS_FILE, 'rb') as f:
            splits_data = pickle.load(f, encoding='latin1')
    except UnicodeDecodeError:
        # Fallback for Python 2 pickle files
        with open(SPLITS_FILE, 'rb') as f:
            splits_data = pickle.load(f, encoding='bytes')
    return splits_data

def load_features_from_pkl(pkl_path, feature_type):
    """Load features from a single .pkl file"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if feature_type == 'bert':
            return data.get('text_bert', np.zeros((1, 768)))[0]  # Remove batch dim
        elif feature_type == 'e5':
            return data.get('text_e5', np.zeros((1, 1024)))[0]  # Remove batch dim
        else:
            return None
    except:
        # Return zero features if file doesn't exist or is corrupted
        if feature_type == 'bert':
            return np.zeros(768)
        elif feature_type == 'e5':
            return np.zeros(1024)
        else:
            return np.zeros(512)  # fallback

def prepare_mustard_data(fold_indices, feature_type, sarcasm_data, identifiers):
    """Prepare MUStARD data using JSON identifiers and labels"""
    X_list = []
    y_list = []
    file_names = []
    
    for split_idx in fold_indices:
        if split_idx >= len(identifiers):
            print(f"Warning: Split index {split_idx} out of range")
            continue
            
        # Get the actual identifier and label from JSON
        identifier = identifiers[split_idx]
        sarcasm_label = sarcasm_data[identifier]['sarcasm']
        
        # Convert boolean to int: True=1 (sarcasm), False=0 (not_sarcasm)  
        label = 1 if sarcasm_label else 0
        
        # Determine which folder to look in based on the label
        if label == 1:  # sarcasm
            context_path = Path(CONTEXT_PATHS['sarcasm']) / f"{identifier}.pkl"
            punchline_path = Path(PUNCHLINE_PATHS['sarcasm']) / f"{identifier}.pkl"
        else:  # not_sarcasm  
            context_path = Path(CONTEXT_PATHS['not_sarcasm']) / f"{identifier}.pkl"
            punchline_path = Path(PUNCHLINE_PATHS['not_sarcasm']) / f"{identifier}.pkl"
        
        # Determine feature extraction method based on feature_type
        if feature_type == 'bert_context_punchline':
            context_feat = load_features_from_pkl(context_path, 'bert')
            punchline_feat = load_features_from_pkl(punchline_path, 'bert')
            combined_feat = np.concatenate([context_feat, punchline_feat])
            
        elif feature_type == 'e5_context_punchline':
            context_feat = load_features_from_pkl(context_path, 'e5')
            punchline_feat = load_features_from_pkl(punchline_path, 'e5')
            combined_feat = np.concatenate([context_feat, punchline_feat])
            
        elif feature_type == 'bert_punchline_only':
            combined_feat = load_features_from_pkl(punchline_path, 'bert')
            
        elif feature_type == 'e5_punchline_only':
            combined_feat = load_features_from_pkl(punchline_path, 'e5')
            
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        X_list.append(combined_feat)
        y_list.append(label)
        file_names.append(identifier)
    
    return np.array(X_list, dtype=np.float32), np.array(y_list), file_names

def train_fold(model, train_loader, val_loader, fold_num, feature_name):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_val = 0.0
    best_epoch_data = None
    
    for epoch in range(EPOCHS):
        model.train()
        for x_b, y_b, _ in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_b.to(device)), y_b.to(device))
            loss.backward()
            optimizer.step()
        
        model.eval()
        epoch_predictions = []
        
        with torch.no_grad():
            for x_b, y_b, fnames in val_loader:
                out = model(x_b.to(device))
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                for i, fname in enumerate(fnames):
                    epoch_predictions.append({
                        'file_name': fname,
                        'true_label': y_b[i].item(),
                        'pred_label': preds[i],
                        'conf_0': probs[i, 0],  # not_sarcasm confidence
                        'conf_1': probs[i, 1],  # sarcasm confidence
                        'fold': fold_num,
                        'epoch': epoch + 1
                    })
        
        trues = [p['true_label'] for p in epoch_predictions]
        preds = [p['pred_label'] for p in epoch_predictions]
        acc = accuracy_score(trues, preds)
        
        if acc > best_val:
            best_val = acc
            best_epoch_data = epoch_predictions
            torch.save(model.state_dict(), f"checkpoints/mustard_text_{feature_name}_fold_{fold_num}.pth")
    
    return best_val, best_epoch_data

# MAIN
splits_data = load_mustard_splits()

# Load sarcasm_data.json to get identifier mapping and labels
with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)

# Create mapping from split index to identifier and label
identifiers = list(sarcasm_data.keys())
print(f"Total samples in JSON: {len(identifiers)}")

for feature_type in TEXT_FEATURES:
    print(f"\n{'='*50}")
    print(f"Training on: TEXT {feature_type.upper()}")
    print(f"{'='*50}")
    
    input_dim = FEATURE_DIMS[feature_type]
    if 'context_punchline' in feature_type:
        base_dim = input_dim // 2
        print(f"Feature dimension: {input_dim} ({base_dim} context + {base_dim} punchline)")
    else:
        print(f"Feature dimension: {input_dim} (punchline only)")
    
    fold_accs = []
    all_fold_predictions = []
    all_preds = []
    all_trues = []
    
    for fold in range(5):  # 0-4 for MUStARD
        print(f"\n--- Fold {fold} ---")
        
        # Get train/test indices for this fold (tuples structure)
        train_indices = splits_data[fold][0]  # First element of tuple
        test_indices = splits_data[fold][1]   # Second element of tuple
        
        # Prepare data
        X_train, y_train, train_fnames = prepare_mustard_data(train_indices, feature_type, sarcasm_data, identifiers)
        X_test, y_test, test_fnames = prepare_mustard_data(test_indices, feature_type, sarcasm_data, identifiers)
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        train_loader = DataLoader(TextDataset(X_train, y_train, train_fnames), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TextDataset(X_test, y_test, test_fnames), batch_size=BATCH_SIZE, shuffle=False)
        
        model = TextModel(input_dim).to(device)
        acc, best_epoch_data = train_fold(model, train_loader, val_loader, fold, feature_type)
        
        print(f"Fold {fold} Best Acc: {acc*100:.2f}%")
        fold_accs.append(acc)
        all_fold_predictions.extend(best_epoch_data)
        
        best_trues = [p['true_label'] for p in best_epoch_data]
        best_preds = [p['pred_label'] for p in best_epoch_data]
        all_preds.extend(best_preds)
        all_trues.extend(best_trues)
    
    predictions_df = pd.DataFrame(all_fold_predictions)
    predictions_df.to_csv(f'{PREDICTIONS_DIR}/mustard_text_{feature_type}_predictions.csv', index=False)
    
    print(f"\n{'='*50}")
    print(f"Results for TEXT {feature_type.upper()}")
    print(f"Average Accuracy: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")
    print(f"{'='*50}")
    print("Confusion Matrix:\n", confusion_matrix(all_trues, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds, target_names=['not_sarcasm', 'sarcasm']))
