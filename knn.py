from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn import datasets
    
class custom_knn():
    def euclidean_distance(self,a,b):
        return distance.euclidean(a,b)

    def fit(self,x,y):
        self.x_train=x
        self.y_train=y

    def predict(self,x_test,k):
        predictions = []
        for x_test_data in x_test:
            label=self.closest(x_test_data,k)
            predictions.append(label)
        return predictions
    
    def closest(self, row,k):
        closest_distance = self.euclidean_distance(row, self.x_train[0])
        closest_index = 0
        distances=[]
        neighbours=[]
        for i in range(1, len(self.x_train)):
            dist = self.euclidean_distance(row, self.x_train[i])
            if dist<closest_distance:
                closest_distance=dist
                closest_index=i
        return self.y_train[closest_index]
k=input("Enter a value for k ")
k=int(k)
iris = datasets.load_iris()
x = iris.data
y = iris.target #labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model=custom_knn()
model.fit(x_train,y_train)
print("Classifier fit. Making Predictions\n")
result=model.predict(x_test,k)
print("Predictions Completed\n")
print("Accuracy:", accuracy_score(y_test, result)*100,"%")
