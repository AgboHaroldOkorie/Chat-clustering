from flask import Flask, render_template, redirect, request, session,json,jsonifym
from flask_mysqldb import MySQL
import pusher
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy
import re
from re import sub

import numpy as np # linear algebra
import pandas as pd
from grade import score 
from gender import*
from remark_student import score


app = Flask(__name__)

app.secret_key = '64ddg74747'
app.config['SESSION_TYPE'] = 'filesystem'

pusher_client = pusher.Pusher(
    app_id="1172263",
    key="c6d6c34f964fb26ee2f6",
    secret="6c79bb89a16988553ebc",
    cluster="mt1"
)

app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "socail_community"

mysql = MySQL(app)



@app.route('/')
def index():
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute('''SELECT * FROM users WHERE username =%s and password = %s ''', (username, password))
        user = cur.fetchone()
        if user:
            session['user'] = {'username': username, 'fullname': user[1]}
            return redirect('chat')
        else:
            msg = 'invalide username or password'
            return render_template('login.html', msg=msg)
    else:
        return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        name = request.form['fullname']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute('''INSERT INTO users (username,password,fullname) VALUES(%s, %s, %s)''', (username, password, name))
        user = mysql.connection.commit()
        cur.close()
        session['user'] = {'username': username, 'fullname': name}
        return redirect('chat')
    else:
        return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


@app.route('/chat')
def chat_app():
    # cur = mysql.connection.cursor()
    # cur.execute('''SELECT * FROM chats''')
    # data = cur.fetchall()
    return render_template('index.html')


@app.route('/chatroom')
def chat_room():
    return render_template('chat_room.html')


@app.route('/send_chat', methods=['POST'])
def send_chat():
    chat = request.form['chat']
    sender = request.form['user_id']
    group_id = request.form['group_id']

    cur = mysql.connection.cursor()
    cur.execute('''INSERT INTO chats (message,user_id,group_id) VALUES(%s, %s, %s)''', (chat, sender, group_id))
    mysql.connection.commit()
    cur.close()
    pusher_client.trigger('chat-channel', 'chat-event', {'message': chat, 'user_id': sender, 'group_id': group_id})
    return jsonify({'result': 'success'})


@app.route('/suggeted_groups')
def suggested_groups():
    cur = mysql.connection.cursor()
    user_chat  = cur.execute('''SELECT * FROM chats WHERE user_id =%s ''', (session['user']['username']))
    #apply data science model

    return jsonify(user_chat)


@app.route('/get_groups/<username>')
def get_groups(username):
    cur = mysql.connection.cursor()
    cur.execute(''' SELECT * FROM groups WHERE user_id= %s ''', [username])
    groups = cur.fetchall()
    return jsonify(groups)



@app.route('/chat/<group_id>')
def get_group(group_id):
    cur = mysql.connection.cursor()
    cur.execute('''SELECT * FROM chats WHERE group_id =%s''', group_id)
    chat = cur.fetchall()
    return render_template('single_chat.html',data=chat)



@app.route('/suggest_group/<user_id>')
def suggest_group(user_id):
    cur = mysql.connection.cursor()

    getchat = '''SELECT * FROM chats WHERE user_id =%s'''
    getgroupd = "SELECT * FROM surgested_groups"

    chat = cur.execute(getchat, [user_id])
    chat = cur.fetchall()

    group = cur.execute(getgroupd)
    group = cur.fetchall()

    words = text_to_word_list(chat)

    # words = words['message']

    recormended = []
    for i in words[1]:
        for dic in group:
            if i == dic:
                a = recormended.append(i)
    return jsonify(recormended)


texts = ["This first text talks about houses and dogs",
        "This is about airplanes and airlines",
        "This is about dogs and houses too, but also about trees",
        "Trees and dogs are main characters in this story",
        "This story is about batman and superman fighting each other",
        "Nothing better than another story talking about airplanes, airlines and birds",
        "Superman defeats batman in the last round"]

# vectorization of the texts
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)
# used words (axis in our multi-dimensional space)
words = vectorizer.get_feature_names()

n_clusters=3
number_of_seeds_to_try=10
max_iter = 300
number_of_process=40 # seads are distributed
model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=number_of_seeds_to_try, n_jobs=number_of_process).fit(X)

labels = model.labels_
# indices of preferible words in each cluster
ordered_words = model.cluster_centers_.argsort()[:, ::-1]

texts_per_cluster = numpy.zeros(n_clusters)
for i_cluster in range(n_clusters):
    for label in labels:
        if label==i_cluster:
            texts_per_cluster[i_cluster] +=1

def prediction(text_to_predict):
    Y = vectorizer.transform([text_to_predict])
    predicted_cluster = model.predict(Y)[0]
    texts_per_cluster[predicted_cluster] += 1

    for term in ordered_words[predicted_cluster, :10]:
        return "\t"+words[term]

def text_to_word_list(text):
    text = str(text)
    text = text.lower()

    # Clean the text
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)

    text = text.split()

    return text  


if __name__ == '__main__':
    app.run(debug=True)
    # socketio.run(app, debug=True)
