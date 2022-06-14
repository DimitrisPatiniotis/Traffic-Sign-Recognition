from sklearn import metrics
from sklearn import svm

class SVM():
    def __init__(self, C=12.5, gamma = 0.5, kernel='rbf'):
        self.svm = svm.NuSVC(nu=0.05, kernel='rbf',gamma=0.00001)
        self.predicitons = []
        self.recall = None
        self.precision = None
        self.f1 = None
        self.accuracy = None
    
    def train(self, train_data, train_labels):
        self.svm.fit(train_data, train_labels)

    def predict(self, sample):
        return self.model.predict(sample)
    
    def make_predictions(self, test_data):
        self.predicitons = self.svm.predict(test_data)

    def get_metrics(self, test_labels):
        self.recall = metrics.recall_score(test_labels, self.predicitons, average='macro')
        self.precision = metrics.precision_score(test_labels, self.predicitons, average='macro')
        self.recall = metrics.f1_score(test_labels, self.predicitons, average='macro')
        self.accuracy = metrics.accuracy_score(test_labels, self.predicitons)

    def run_svm_experiment(self, training_data, training_labels, test_data, test_labels):
        self.train(training_data, training_labels)
        self.make_predictions(test_data)
        self.get_metrics(test_labels)

if __name__ == '__main__':
    print('ML Baseline Process')