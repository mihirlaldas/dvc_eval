from pickle import load

from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
import json
import os


model = load(open('model/model.pkl', 'rb'))
print(model.predict(['bone lasted forever, will buy again']))
print(model.predict(['nice lipstick']))
print(model.predict(['wipes are fragrant']))

original_x = load(open('data/prepared/original_x.pkl', 'rb'))
original_y = load(open('data/prepared/original_y.pkl', 'rb'))

y_pred = model.predict(original_x)
report = classification_report(original_y, y_pred, output_dict=True)
print(classification_report(original_y, y_pred))
os.makedirs("report", exist_ok=True)

with open('report/classification.json', 'w') as f:
    json.dump(report, f)
plot_confusion_matrix(model,original_y, y_pred)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('report/confusion_matrix.png')
plt.show()