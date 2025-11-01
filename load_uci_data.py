import pandas as pd

def load_promoters_data():
    # Read the .data file - common UCI format
    data = []
    with open('promoters.data', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(line)
    
    # Process data
    sequences = []
    labels = []
    
    for line in data:
        parts = line.split(',')
        if len(parts) >= 2:
            label = parts[0].strip()
            sequence = ''.join(parts[1:]).replace(' ', '').replace("'", "")
            
            sequences.append(sequence)
            labels.append(1 if label == '+' else 0)
    
    # Create dataset
    df = pd.DataFrame({'sequence': sequences, 'label': labels})
    df.to_csv('dna_promoter_dataset.csv', index=False)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Promoters: {sum(labels)}, Non-promoters: {len(labels)-sum(labels)}")
    return df

if __name__ == "__main__":
    load_promoters_data()