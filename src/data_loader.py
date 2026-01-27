import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and preprocess the Heart Failure dataset."""
    
    def __init__(self, filepath='data/heart_failure_clinical_records_dataset.csv'):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Loaded {self.df.shape[0]} records")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found: {self.filepath}")

    def preprocess(self):
        if self.df is None:
            self.load_data()
        
        # Remove duplicates if any
        n_dup = self.df.duplicated().sum()
        if n_dup > 0:
            print(f"Dropped {n_dup} duplicates")
            self.df = self.df.drop_duplicates()
        
        return self.df

    def get_split_data(self, test_size=0.2, random_state=42):
        """Split into train/test with stratification."""
        if self.df is None:
            self.preprocess()

        X = self.df.drop('DEATH_EVENT', axis=1)
        y = self.df['DEATH_EVENT']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.get_split_data()
