import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

def neural_network_model(x, data, hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer):

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

def use_neural_network(x, input_data, hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer, saver, lemmatizer):
    print('X is:', x)
    prediction = neural_network_model(x, input_data,hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer)
    with open('NLP/lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"NLP/model.ckpt")
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

def test_neural_network(x, hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer, saver, lemmatizer):
    print(use_neural_network(x,"He's an idiot and a jerk.", hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer, saver, lemmatizer))
    print(use_neural_network(x,"This was the best store i've ever seen.", hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer, saver, lemmatizer))
    print(use_neural_network(x,"I am happy.", hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer, saver, lemmatizer))
    # print(use_neural_network(x,"Today was a good day."))
    # print(use_neural_network(x,"Today was a bad day."))
    # print(use_neural_network(x,"This is bad :)"))
    # print(use_neural_network(x,"This is good :)"))
    # print(use_neural_network(x,"I am sad."))
    # print(use_neural_network(x,"Happy happy happy :(."))
    # print(use_neural_network(x,"Hello :("))
    # print(use_neural_network(x,"Unknown :("))
    # print(use_neural_network(x,"I convinced myself that my sadness was like a layer of skin I could never be rid of, but I realised that I’m not ready to give up just yet. There is still some fight left in me, and though the tunnel seems long and dark, there has to be the famed light at the end of it, and I want to find a way back to the light, for my parents and the people who still care for me, but most of all, for myself."))
    # print(use_neural_network(x,"If someone were to ask me what depression feels like, I would say it feels like being 10 feet underwater with your feet tied to an anchor that’s pulling you down, down, down. You know you need to find your way back to the surface but you can’t seem to untie yourself. Will-power has reduced to nothingness."))
    # print(use_neural_network(x,"Today started off with a bang"))
    # print(use_neural_network(x,"Literally"))
    # print(use_neural_network(x,"I woke up to the sound of a giant pillar falling in the construction site beside my room"))
    # print(use_neural_network(x,"What the fuck"))
    # print(use_neural_network(x,"These next few months living here are gonna be horrible"))
    # print(use_neural_network(x,"It didn’t help that I only got 5 hours of sleep either, and I spilled my cereal on my shirt because of my grogginess"))
    # print(use_neural_network(x,"At least when I got to class my best friend told me that he was going to give me $500 because he won the lottery"))
    # print(use_neural_network(x,"Class was just the usual, and I went to the gym for the first time in 2 weeks"))
    # print(use_neural_network(x,"As expected, my max weights dropped another 10 pounds…oops"))
    # print(use_neural_network(x,"Anyway, I skipped the club meeting I had because I was feeling lazy, and here I am, typing this diary because I got nothing else done today"))
    # print(use_neural_network(x,"@stellargirl I loooooooovvvvvveee my Kindle2. Not that the DX is cool, but the 2 is fantastic in its own right."))
    # print(use_neural_network(x,"Fuck this economy. I hate aig and their non loan given asses."))
    # print(use_neural_network(x,"Jquery is my new best friend."))
    # print(use_neural_network(x,"how can you not love Obama? he makes jokes about himself."))
    # print(use_neural_network(x,"@Karoli I firmly believe that Obama/Pelosi have ZERO desire to be civil.  It's a charade and a slogan, but they want to destroy conservatism"))
    # print(use_neural_network(x," "))
    # print(use_neural_network(x," ") ) 
def printHi(var):
    print(var)
