import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import cv2

app = Flask(__name__)
X = pickle.load(open("dataX.pkl", "rb"))
vectorizer = pickle.load(open("dataVectorizer.pkl", "rb"))
textClean = pickle.load(open("dataClean.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/cluster", methods=["POST"])
def cluster():
    int_cluster = int(request.form['clusterNum'])
    # fitting model
    kmeans = KMeans(int_cluster, random_state=161).fit(X)
    # melakukan sort pada vektor centroid
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(int_cluster):
        clusterId = np.where(kmeans.labels_ == i)[0].tolist()
        clusterText = "".join([textClean[i] for i in clusterId])
        plt.figure(figsize=(8, 10))  # ngeset ukuran gambar
        wf = WordCloud(background_color='white', max_words=1000,
                       random_state=113).generate(clusterText)
        plt.imshow(wf)
        # plt.show()
        wf.to_file("static/word{}.png".format(i+1))
    Maks = 20
    Sisa = Maks-int_cluster
    print(Sisa)
    for i in range(Sisa):
        whiteblankimage = 255 * np.ones(shape=[200, 100, 3], dtype=np.uint8)
        cv2.imwrite('static/word{}.png'.format(int_cluster+i+1),
                    whiteblankimage)
    return render_template('index.html', lok="word{}.png".format(i))


if __name__ == "__main__":
    app.run(debug=True)
