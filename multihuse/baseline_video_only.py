import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# CONFIGURATION
SPLITS_DIR = r'multihuse_dataset\splits_strict_v2'
FEATURES_DIR = r'multihuse_dataset'

VIDEO_FEATURES = [
    'pe_core_visual_embeddings.csv',
    'videomae_visual_embeddings.csv'
]

LABEL_COLUMN = 'humor_style'
NUM_CLASSES = 5
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

class VideoDataset(Dataset):
    def __init__(self, x, y, file_names): 
        self.x, self.y, self.file_names = x, y, file_names
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, i): 
        return torch.tensor(self.x[i]), torch.tensor(self.y[i], dtype=torch.long), self.file_names[i]

class VideoModel(nn.Module):
    def __init__(self, input_dim):
        super(VideoModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
    def forward(self, x):
        return self.net(x)

def load_fold_split(fold_num):
    train_split = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_train.csv')
    test_split = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_test.csv')
    return train_split['file_name'].values, test_split['file_name'].values

def prepare_data(features_df, file_names_subset, le=None):
    df_subset = features_df[features_df['file_name'].isin(file_names_subset)].copy()
    df_subset = df_subset.sort_values('file_name').reset_index(drop=True)
    
    feat_cols = sorted([c for c in df_subset.columns if c.startswith('visual_feat_')])
    X = df_subset[feat_cols].values.astype(np.float32)
    file_names = df_subset['file_name'].values
    
    if le is None:
        le = LabelEncoder()
        y = le.fit_transform(df_subset[LABEL_COLUMN].values)
        return X, y, file_names, le
    else:
        y = le.transform(df_subset[LABEL_COLUMN].values)
        return X, y, file_names

def train_fold(model, train_loader, val_loader, fold_num, feature_name):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch_data = None
    
    for epoch in range(EPOCHS):
        model.train()
        for v_batch, l_batch, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(v_batch.to(device))
            loss = criterion(outputs, l_batch.to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        epoch_predictions = []
        
        with torch.no_grad():
            for v_batch, l_batch, fnames in val_loader:
                logits = model(v_batch.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                for i, fname in enumerate(fnames):
                    epoch_predictions.append({
                        'file_name': fname,
                        'true_label': l_batch[i].item(),
                        'pred_label': preds[i],
                        'conf_0': probs[i, 0],
                        'conf_1': probs[i, 1],
                        'conf_2': probs[i, 2],
                        'conf_3': probs[i, 3],
                        'conf_4': probs[i, 4],
                        'fold': fold_num,
                        'epoch': epoch + 1
                    })
        
        trues = [p['true_label'] for p in epoch_predictions]
        preds = [p['pred_label'] for p in epoch_predictions]
        acc = accuracy_score(trues, preds)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch_data = epoch_predictions
            torch.save(model.state_dict(), f"checkpoints/video_{feature_name}_fold_{fold_num}.pth")
            
    return best_acc, best_epoch_data

# MAIN
for feature_file in VIDEO_FEATURES:
    feature_name = feature_file.replace('.csv', '')
    print(f"\n{'='*50}")
    print(f"Training on: {feature_name}")
    print(f"{'='*50}")
    
    features_df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}')
    
    # Get input dimension from features
    feat_cols = [c for c in features_df.columns if c.startswith('visual_feat_')]
    input_dim = len(feat_cols)
    print(f"Feature dimension: {input_dim}")
    
    fold_accs = []
    all_fold_predictions = []
    all_preds = []
    all_trues = []
    
    for fold in range(1, 6):
        print(f"\n--- Fold {fold} ---")
        train_files, test_files = load_fold_split(fold)
        
        X_train, y_train, train_fnames, le = prepare_data(features_df, train_files)
        X_test, y_test, test_fnames = prepare_data(features_df, test_files, le)
        
        train_loader = DataLoader(VideoDataset(X_train, y_train, train_fnames), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(VideoDataset(X_test, y_test, test_fnames), batch_size=BATCH_SIZE, shuffle=False)
        
        model = VideoModel(input_dim).to(device)
        acc, best_epoch_data = train_fold(model, train_loader, val_loader, fold, feature_name)
        
        print(f"Fold {fold} Best Acc: {acc*100:.2f}%")
        fold_accs.append(acc)
        all_fold_predictions.extend(best_epoch_data)
        
        best_trues = [p['true_label'] for p in best_epoch_data]
        best_preds = [p['pred_label'] for p in best_epoch_data]
        all_preds.extend(best_preds)
        all_trues.extend(best_trues)
    
    predictions_df = pd.DataFrame(all_fold_predictions)
    predictions_df.to_csv(f'{PREDICTIONS_DIR}/video_{feature_name}_predictions.csv', index=False)
    
    print(f"\n{'='*50}")
    print(f"Results for {feature_name}")
    print(f"Average Accuracy: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")
    print(f"{'='*50}")
    print("Confusion Matrix:\n", confusion_matrix(all_trues, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds))