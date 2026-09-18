#!/usr/bin/env python3
"""
MUStARD Sequential Feature Extraction
-------------------------------------------------------------------
PE-Core, E5-multilingual, and Dasheng implementations.

"""

import os
import sys
import logging
import pickle
import numpy as np
import torch
import gc
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

# --- Imports ---
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, VideoMAEModel, AutoFeatureExtractor
from sentence_transformers import SentenceTransformer
import torchaudio
from torchvision import transforms as T

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# --- PE Import ---
try:
    import core.vision_encoder.pe as pe
    import mediapipe as mp
    PE_AVAILABLE = True
except ImportError:
    PE_AVAILABLE = False

# --- Silence Warnings ---
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

class FacePreProcessor:
    """Helper to detect and crop faces for PE."""
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def get_face_crops(self, video_path, num_frames=5):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return []

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        face_images = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)
            
            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                pad_w, pad_h = int(w * 0.1), int(h * 0.1)
                x, y = max(0, x - pad_w), max(0, y - pad_h)
                w, h = min(iw - x, w + 2*pad_w), min(ih - y, h + 2*pad_h)
                
                face_crop = frame_rgb[y:y+h, x:x+w]
                if face_crop.size == 0: face_img = Image.fromarray(frame_rgb)
                else: face_img = Image.fromarray(face_crop)
            else:
                face_img = Image.fromarray(frame_rgb)
            
            face_images.append(face_img)
            
        cap.release()
        return face_images

class MustardSequentialExtractor:
    def __init__(self, input_dir: str, output_dir: str, device: str = "cuda"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.setup_logging()
        
        # Identify all target files once
        self.logger.info("Indexing files...")
        self.files = []
        for video_path in self.input_dir.rglob("*.mp4"):
            rel_path = video_path.relative_to(self.input_dir)
            out_path = self.output_dir / rel_path.with_suffix('.pkl')
            self.files.append({
                'video': video_path,
                'audio': video_path.with_suffix('.wav'),
                'text': video_path.with_suffix('.txt'),
                'out': out_path
            })
        self.logger.info(f"Found {len(self.files)} files to process.")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
        self.logger = logging.getLogger(__name__)

    def save_features(self, out_path, new_features):
        """Helper to update existing pickle file or create new one."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            try:
                with open(out_path, 'rb') as f:
                    data = pickle.load(f)
            except:
                data = {}
        else:
            data = {}
        
        data.update(new_features)
        
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)

    def clear_gpu(self):
        """Force clean GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()

    # ========================== STAGE 1: TEXT ==========================
    def process_text(self):
        self.logger.info("--- STAGE 1: TEXT EXTRACTION (BERT + E5) ---")
        self.clear_gpu()
        
        # Load Models
        bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device).eval()
        
        e5_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct').to(self.device).eval()
        
        for item in tqdm(self.files, desc="Text"):
            feats = {}
            try:
                text = ""
                if item['text'].exists():
                    with open(item['text'], 'r', errors='ignore') as f: text = f.read().strip()
                
                # BERT
                inputs = bert_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                with torch.no_grad(): feats['text_bert'] = bert_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()

                with torch.no_grad():
                    prefixed_text = f"query: {text}"  
                    feats['text_e5'] = e5_model.encode([prefixed_text], convert_to_tensor=True, device=self.device).cpu().numpy()
            except Exception as e:
                self.logger.warning(f"Text processing failed: {e}")
                feats['text_bert'] = np.zeros((1, 768))
                feats['text_e5'] = np.zeros((1, 1024))
            
            self.save_features(item['out'], feats)

        # Cleanup
        del bert_model, bert_tok, e5_model
        self.clear_gpu()

    # ========================== STAGE 2: AUDIO ==========================
    def process_audio(self):
        self.logger.info("--- STAGE 2: AUDIO EXTRACTION (Dasheng) ---")
        self.clear_gpu()

        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained("mispeech/dasheng-0.6B", trust_remote_code=True)
            model = AutoModel.from_pretrained("mispeech/dasheng-0.6B", outputdim=None, trust_remote_code=True).to(self.device).eval()
        except Exception as e:
            self.logger.error(f"Dasheng Load Failed: {e}")
            return

        for item in tqdm(self.files, desc="Audio"):
            feats = {}
            try:
                if item['audio'].exists():
                    speech, sr = torchaudio.load(str(item['audio']))
                    if sr != 16000: speech = torchaudio.transforms.Resample(sr, 16000)(speech)
                    if speech.shape[0] > 1: speech = torch.mean(speech, dim=0, keepdim=True)
                    
                    audio_array = speech.squeeze().numpy()
                    inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
                    
                    # Move all inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)  
                        feats['audio_dasheng'] = outputs.logits.cpu().numpy()  # Shape: (1, 1280)
                else:
                    feats['audio_dasheng'] = np.zeros((1, 1280))
            except Exception as e:
                self.logger.warning(f"Audio processing failed: {e}")
                feats['audio_dasheng'] = np.zeros((1, 1280))

            self.save_features(item['out'], feats)

        del model, feature_extractor
        self.clear_gpu()

    # ========================== STAGE 3: VIDEO ==========================
    def process_video(self):
        self.logger.info("--- STAGE 3: VIDEO EXTRACTION (MAE + PE) ---")
        self.clear_gpu()

        # Load MAE
        mae_proc = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        mae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(self.device).eval()

        # Load PE
        pe_model = None
        face_proc = None
        pe_transform = None

        if PE_AVAILABLE:
            try:
                face_proc = FacePreProcessor()
                if hasattr(pe, 'CLIP'): pe_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
                elif hasattr(pe, 'create_model'): pe_model = pe.create_model("PE-Core-L14-336", pretrained=True)
                else: raise AttributeError("PE Create failed")
                
                pe_model.to(self.device).eval()
                
                pe_transform = T.Compose([
                    T.Resize((224, 224), interpolation=BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
                
                self.logger.info("PE-Core model loaded successfully")
            except Exception as e:
                self.logger.error(f"PE Load Failed: {e}")
                pe_model = None

        def get_mae_frames(path):
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0: return [np.zeros((224,224,3), dtype=np.uint8)] * 16
            indices = np.linspace(0, total-1, 16, dtype=int)
            frames = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, f = cap.read()
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ret else np.zeros((224,224,3), dtype=np.uint8))
            cap.release()
            while len(frames) < 16: frames.append(frames[-1])
            return frames[:16]

        for item in tqdm(self.files, desc="Video"):
            feats = {}
            
            # MAE
            try:
                frames = get_mae_frames(item['video'])
                inputs = mae_proc(list(frames), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feats['video_mae'] = torch.mean(mae_model(**inputs).last_hidden_state, dim=1).cpu().numpy()
            except Exception as e:
                self.logger.warning(f"MAE processing failed: {e}")
                feats['video_mae'] = np.zeros((1, 768))

            # model calling and tuple unpacking
            if pe_model:
                try:
                    faces = face_proc.get_face_crops(item['video'], 5)
                    if faces:
                        frame_embeddings = []
                        for face_img in faces:
                            # Process single image
                            processed_image = pe_transform(face_img).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                image_features, _, _ = pe_model(processed_image, None)
                                frame_embeddings.append(image_features)
                        
                        # Average across frames
                        if frame_embeddings:
                            avg_embedding = torch.mean(torch.cat(frame_embeddings, dim=0), dim=0, keepdim=True)
                            feats['video_pe'] = avg_embedding.cpu().numpy()
                        else:
                            feats['video_pe'] = np.zeros((1, 1024))
                    else:
                        feats['video_pe'] = np.zeros((1, 1024))
                except Exception as e:
                    self.logger.warning(f"PE processing failed: {e}")
                    feats['video_pe'] = np.zeros((1, 1024))
            else:
                feats['video_pe'] = np.zeros((1, 1024))

            self.save_features(item['out'], feats)

        del mae_model, pe_model
        self.clear_gpu()

    def run(self):
        self.logger.info("🚀 STARTING FEATURE EXTRACTION")
        self.process_text()
        self.process_audio()
        self.process_video()
        self.logger.info("✅ ALL STAGES COMPLETE")

def main():
    # UPDATE PATHS
    INPUT_DIR = "mustard_dataandcode/mustard_organized_dataset"
    OUTPUT_DIR = "mustard_dataandcode/mustard_features_pkl_FIXED" 
    
    extractor = MustardSequentialExtractor(INPUT_DIR, OUTPUT_DIR)
    extractor.run()

if __name__ == "__main__":
    main()
