
import seaborn as sns
import pandas as pd
import sys
import os

if (len(sys.argv))<2:
    raise TypeError('Not Enough Input Arguments')

data_path = sys.argv[1]

results = []
files = os.listdir(data_path)
for file in files:
    if file.find('summary.csv') > 0:
        info = file.split('_')
        data = info[0]
        norm = info[1]
        classifier = info[3]
        num_clusters = info[4]
        file = data_path + file
        summary = pd.DataFrame.from_csv(file, index_col=0)
        acc = summary['Accuracy'].mean()
        results.append([data,norm,classifier,num_clusters, acc])

results = pd.DataFrame(results, columns=['Dataset', 'Normalization Method', 'Classifier', 'k', 'Accuracy'])
results = results.pivot_table(values=['Accuracy'], columns= ['Classifier'], index=['Dataset', 'Normalization Method', 'k'])
cm = sns.light_palette("green", as_cmap=True)
s = results.style.background_gradient(cmap=cm)

with open('classification_results_summary.html', 'w') as html_file:
    html_file.write(s.render())
