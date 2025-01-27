import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(url):
    """
    Load and preprocess the wine dataset.

    Parameters:
        url (str): The URL to the dataset.

    Returns:
        tuple: Training and testing datasets (X_train, X_test, y_train, y_test)
    """
    # Define column names
    wine_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                  'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                  'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
    
    # Load the dataset
    df = pd.read_csv(url, names=wine_names)

    # Separate features and target
    X = df[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 
            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 
            'Hue', 'OD280/OD315', 'Proline']]
    y = df['Class']

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

# Usage
if __name__=="__main__":
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(url)
    print("Data loaded and split successfully!")