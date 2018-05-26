import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
hm_data = 2000000

batch_size = 32
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')


current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weights':tf.Variable(tf.random_normal([205, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
                  'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}

output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
                'biases': tf.Variable(tf.constant(0.1, shape=[n_classes])), }


def neural_network_model(data):

    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # now goes through an activation function - sigmoid function
    layer_1 = tf.nn.relu(layer_1)
    # input for layer 2 = result of activ_func for layer 1
    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']

    return output

saver = tf.train.import_meta_graph('./model.ckpt.meta')

def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"./model.ckpt")
        # import the inspect_checkpoint library
        from tensorflow.python.tools import inspect_checkpoint as chkp

        # print all tensors in checkpoint file
        #chkp.print_tensors_in_checkpoint_file("./model.ckpt", tensor_name='', all_tensors=True)
        #saver.restore(sess,tf.train.latest_checkpoint('./'))

        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        test = prediction.eval(feed_dict={x:[features]})
        print(test)
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        if result[0] == 0:
            print('Positive:',input_data)
        elif result[0] == 1:
            print('Negative:',input_data)

use_neural_network("He's an idiot and a jerk.")
use_neural_network("This was the best store i've ever seen.")
use_neural_network("I am happy.")
use_neural_network("Today was a good day.")
use_neural_network("Today was a bad day.")
use_neural_network("This is bad :)")
use_neural_network("This is good :)")
use_neural_network("I am sad.")
use_neural_network("Happy happy happy :(.")
use_neural_network("Hello :(")
use_neural_network("Unknown :(")
use_neural_network("I convinced myself that my sadness was like a layer of skin I could never be rid of, but I realised that I’m not ready to give up just yet. There is still some fight left in me, and though the tunnel seems long and dark, there has to be the famed light at the end of it, and I want to find a way back to the light, for my parents and the people who still care for me, but most of all, for myself.")
use_neural_network("If someone were to ask me what depression feels like, I would say it feels like being 10 feet underwater with your feet tied to an anchor that’s pulling you down, down, down. You know you need to find your way back to the surface but you can’t seem to untie yourself. Will-power has reduced to nothingness.")
use_neural_network("Today started off with a bang")
use_neural_network("Literally")
use_neural_network("I woke up to the sound of a giant pillar falling in the construction site beside my room")
use_neural_network("What the fuck")
use_neural_network("These next few months living here are gonna be horrible")
use_neural_network("It didn’t help that I only got 5 hours of sleep either, and I spilled my cereal on my shirt because of my grogginess")
use_neural_network("At least when I got to class my best friend told me that he was going to give me $500 because he won the lottery")
use_neural_network("Class was just the usual, and I went to the gym for the first time in 2 weeks")
use_neural_network("As expected, my max weights dropped another 10 pounds…oops")
use_neural_network("Anyway, I skipped the club meeting I had because I was feeling lazy, and here I am, typing this diary because I got nothing else done today")
use_neural_network("@stellargirl I loooooooovvvvvveee my Kindle2. Not that the DX is cool, but the 2 is fantastic in its own right.")
use_neural_network("Fuck this economy. I hate aig and their non loan given asses.")
use_neural_network("Jquery is my new best friend.")
use_neural_network("how can you not love Obama? he makes jokes about himself.")
use_neural_network("@Karoli I firmly believe that Obama/Pelosi have ZERO desire to be civil.  It's a charade and a slogan, but they want to destroy conservatism")
use_neural_network(" ")
use_neural_network(" ")