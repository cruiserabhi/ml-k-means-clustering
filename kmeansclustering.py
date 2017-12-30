import json
import csv
import tempfile
from flask import Flask, render_template, redirect, url_for, request, make_response,jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
import plotly
import copy
import plotly.plotly as py
import plotly.graph_objs as go

app = Flask(__name__)
@app.route('/')
def form():
	return render_template('index3.html')
		
@app.route('/upload', methods=["POST"])
def transform_view():
	file = request.files['data_file']
	if request.method == 'POST':
		tempfile_path = tempfile.NamedTemporaryFile().name
		file.save(tempfile_path)
		dataset = pd.read_csv(tempfile_path )
		X = dataset.iloc[:,:].values

		mv=""
		if 'Column Containing Missing Values' in request.form:
			mv=request.form['Column Containing Missing Values']
		else:
			mv=""
		cdv=request.form['Column Containing Categorical Values']
		if mv is not "" :
			mv1=mv.split(',')
			for i in range(0 , len(mv1)):
				elem = int(mv1[i])
				d=elem+1
				imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
				imputer=imputer.fit(X[:,elem:d])
				X[:,elem:d]=imputer.transform(X[:,elem:d])
			#print(X)
			#sys.stdout.flush()
			#print(Y)
			#sys.stdout.flush()
		if cdv is not "" :
			c=0
			#print(X_org_test)	
			#sys.stdout.flush()
			#print("Printing CDV:\n")
			#sys.stdout.flush()
			split_cdv = cdv.split(',')
			#print(split_cdv)
			#sys.stdout.flush()
			mv1=[];			
			for i in range(0 , len(split_cdv)):
				elem = int(split_cdv[i])
				mv1.append(elem)
				labelencoder_X = LabelEncoder()
				X[:,elem] = labelencoder_X.fit_transform(X[:,elem])
				#print(X)
				#sys.stdout.flush()
				#print('Label Encoding Done')
				#sys.stdout.flush()
			onehotencoder=OneHotEncoder(categorical_features=mv1)
			X=onehotencoder.fit_transform(X).toarray()
			#print('One Hot Encoding Done')

		

		sc = StandardScaler()
		X = sc.fit_transform(X)

		pca = PCA(n_components = 2)
		X = pca.fit_transform(X)
		explained_variance = pca.explained_variance_ratio_		
			
		wcss = []
		for i in range(1,11) :
			kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
			kmeans.fit(X)
			wcss.append(kmeans.inertia_)
		graphs = [ 
			dict(
				data=[
					dict(
						x=[1,2,3,4,5,6,7,8,9,10],
						y=wcss,
						type='scatter'
					),
				],
			
		layout = dict(
			title='Elbow Method Graph',
			yaxis=dict(
						title='WCSS'
						),
			xaxis=dict( 
						title='No. of Clusters'
						)
					)
				)	
			]	
		print(X)
		ids=['graph-{}'.format(i) for i, _ in enumerate(graphs)]
		graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)	
		return render_template("elbow.html",ids=ids,graphJSON=graphJSON)






@app.route('/result', methods=["POST"])		
def view():
	file = request.files['data_file']
	if request.method == 'POST':
		tempfile_path = tempfile.NamedTemporaryFile().name
		file.save(tempfile_path)
		col=pd.read_csv(tempfile_path, nrows=1).columns.tolist()
		col1=str(col[-2]) 
		col2=str(col[-1])
		col=col[-2:]
		dataset = pd.read_csv(tempfile_path)
		X = dataset.iloc[:,:].values
		n=int(request.form['nclusters'])
		print(n)
		print(X)
		
		mv=""
		if 'Column Containing Missing Values' in request.form:
			mv=request.form['Column Containing Missing Values']
		else:
			mv=""
		cdv=request.form['Column Containing Categorical Values']
		if mv is not "" :
			mv1=mv.split(',')
			for i in range(0 , len(mv1)):
				elem = int(mv1[i])
				d=elem+1
				imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
				imputer=imputer.fit(X[:,elem:d])
				X[:,elem:d]=imputer.transform(X[:,elem:d])
			#print(X)
			#sys.stdout.flush()
			#print(Y)
			#sys.stdout.flush()
		if cdv is not "" :
			c=0
			#print(X_org_test)	
			#sys.stdout.flush()
			#print("Printing CDV:\n")
			#sys.stdout.flush()
			split_cdv = cdv.split(',')
			#print(split_cdv)
			#sys.stdout.flush()
			mv1=[];			
			for i in range(0 , len(split_cdv)):
				elem = int(split_cdv[i])
				mv1.append(elem)
				labelencoder_X = LabelEncoder()
				X[:,elem] = labelencoder_X.fit_transform(X[:,elem])
				#print(X)
				#sys.stdout.flush()
				#print('Label Encoding Done')
				#sys.stdout.flush()
			onehotencoder=OneHotEncoder(categorical_features=mv1)
			X=onehotencoder.fit_transform(X).toarray()
			#print('One Hot Encoding Done')

		

		sc = StandardScaler()
		X = sc.fit_transform(X)

		pca = PCA(n_components = 2)
		X = pca.fit_transform(X)
		explained_variance = pca.explained_variance_ratio_		

		
		kmeans = KMeans(n_clusters=n,init='k-means++',max_iter=300,n_init=10,random_state=0)
		kmeans.fit(X)
		y_kmeans = kmeans.predict(X)
		print(y_kmeans)
		pred=[]
		for i in range(0,len(y_kmeans)) :
			pre=float(y_kmeans[i])
			pred.append(pre)
		df=pd.DataFrame(X,columns=col)	
		df['Clusters']=pd.Series(pred,dataset.index)
		dataset['Clusters']=pd.Series(pred,dataset.index)
		print(dataset)
		c=[] #list for x-axis
		d=[] #list for y-axis
		data = []
		layout = dict(
			title='K-Means Clustering for Clients',
			yaxis=dict(
						title='DImension-1'
						),
			xaxis=dict( 
						title='Dimension-2'
						)
					)		
		index=len(dataset.index)
		red = 152
		alpha = .8		
		for i in range(0,n): #n=Total Number of Clusters
			for j in range(0,index) : #index=Number of Attributes
				cluster=df.iloc[j][2]
				attr1=df.iloc[j][0]
				attr2=df.iloc[j][1]
				if cluster==i :
					c.append(attr1)	#C Contains all attributes corresponding to Cluster i (X-Axis)
					d.append(attr2) #D Contains all attributes corresponding to Cluster i (Y-Axis)
			red = red + 10
			alpha = alpha - .1
			trace0 = go.Scatter(
				x = copy.copy(c),
				y = copy.copy(d),
				name = 'Cluster-' + str(i),
				mode = 'markers',
				marker = dict(
					size = 10,
					color = 'rgba(red, 0, 0, alpha)',
					line = dict(
						width = 2,
						color = 'rgb(0, 0, 0)'
					)
				)
			)
			data.append(trace0)
			c[:] = []
			d[:] = []
		graphs = [dict(data=data,layout=layout)]	
		"""graphs = [ 
				dict(
					data=[
						dict(
							x=c,
							y=d,
							mode='markers',
							
						),
					],
			layout = dict(
				title='K-Means Clustering for Clients',
				yaxis=dict(
							title=col2
							),
				xaxis=dict( 
							title=col1
							)
						)
					)	
				]"""		
		ids=['graph-{}'.format(i) for i, _ in enumerate(graphs)]
		graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)	
		return render_template("res.html",ids=ids,graphJSON=graphJSON,res=[dataset.to_html(classes='sheet')])	
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080)		
