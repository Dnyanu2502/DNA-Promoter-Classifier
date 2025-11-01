import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class DNAPromoterPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.nucleotides = 'ATCG'
    
    def extract_features(self, sequence):
        sequence = sequence.upper()
        features = []
        features.append(len(sequence))
        for nt in self.nucleotides:
            features.append(sequence.count(nt) / len(sequence))
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        features.append(gc_content)
        return features
    
    def train(self, sequences, labels):
        X = [self.extract_features(seq) for seq in sequences]
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        return accuracy
    
    def predict(self, sequences):
        features = [self.extract_features(seq) for seq in sequences]
        probabilities = self.model.predict_proba(features)
        predictions = self.model.predict(features)
        results = []
        for seq, pred, prob in zip(sequences, predictions, probabilities):
            results.append({
                'sequence': seq,
                'prediction': 'Promoter' if pred == 1 else 'Non-Promoter',
                'confidence': max(prob),
                'promoter_probability': prob[1]
            })
        return results

def create_and_save_model():
    df = pd.read_csv('dna_promoter_dataset.csv')
    predictor = DNAPromoterPredictor()
    accuracy = predictor.train(df['sequence'].tolist(), df['label'].tolist())
    joblib.dump(predictor, 'dna_promoter_model.joblib')
    print("Model saved!")
    return accuracy

if __name__ == "__main__":
    create_and_save_model()