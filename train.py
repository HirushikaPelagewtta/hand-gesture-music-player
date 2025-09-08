import pandas as pd, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/hand_samples.csv"
MODEL_PATH = "models/gesture_svm.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    y, X = df['label'], df.drop(columns=['label'])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    clf = Pipeline([('scaler',StandardScaler()),('svm',SVC(kernel='rbf',probability=True))])
    clf.fit(Xtr,ytr)
    yp = clf.predict(Xte)
    print("Accuracy:", accuracy_score(yte,yp))
    print(classification_report(yte,yp))
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

if __name__=="__main__": main()
