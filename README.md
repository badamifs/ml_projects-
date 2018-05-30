# machine learning projects
This repo is just me practicing data clean ups and modelling different data types into various ml models

Steps taken:
Create a anaconda environment for my machine learning adventures:

``conda create -n ml_projects python=2.7``

Install required packages:

``pip install <package_name>``

I and playing around with 2 different text editors

sublime: https://www.sublimetext.com/

atom: https://atom.io/

atom has some cool features and github integrations. Highly recommend. 

Next, I type up my code and convert it to a jupyter notebook for a more interactive display. you can also directly start with a notebook but I prefer this conversion method:

``git clone https://github.com/sklam/py2nb.git``

Installation:

``cd py2nb``

``python setup.py install``

Usage:

``python -m py2nb input.py output.ipynb``

# Basic ml
check out the notebook basic_ml.ipynb. This gives you a basic framework for splitting, visualizing and predicting. You can use this framework with different datsets and test which algorith works best for your dataset.

# Anomaly detection
This technique works well when you know most of your dataset is negative (i.e. does not meet anomaly criteria) and only has a small subset of postive outcomes (i.e. meet anomaly criteria)

In the code I use a simple moving average approach to get trends and then mark anything 3 standard deviations away from the moving avergage as an anomaly

Very useful with time series data. I visualized using a simple plot funtion and a lag_plot within the panda library. The lag plot does a good job at detecting outliers as well. 