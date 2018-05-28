"""Filename: server.py
"""
from __future__ import print_function
from apiclient import discovery
from httplib2 import Http
from oauth2client import file, client, tools
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import json

from flask_cors import CORS
import os
import re
# import pandas as pd
#from sklearn.externals import joblib
from flask import Flask, jsonify, request
import dill as pickle
#import builtins
#from sklearn.feature_extraction.text import CountVectorizer
import json
import re

import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])

def apicall():

    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        # test_json = request.get_json()
        # test = pd.read_json(test_json, orient='records')

        # #To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
        # test['Dependents'] = [str(x) for x in list(test['Dependents'])]

        # #Getting the Loan_IDs separated out
        # loan_ids = test['Loan_ID']
        textToAnalyze_json = request.get_json()
        textToAnalyzeWhole = json.loads(json.dumps(textToAnalyze_json['text']))
        textToAnalyze = json.loads(json.dumps(textToAnalyze_json['text']))
        #Split text into array of sentences
        #textToAnalyze = textToAnalyze.split('.')
        #   regular expressions are easiest (and fastest)
        sentenceEnders = re.compile('[.!?]')
        textToAnalyze = sentenceEnders.split(textToAnalyze)
        #print(textToAnalyze)

    except Exception as e:
        raise e

    clf = 'emotion_classifier.pk'
    if textToAnalyze == "":
        return(bad_request())
    else:

        #Load the count vectorizer
        
        #print("Loading the count vectorizer...")
        count_vect = None
        with open('models/countVectorizer.pk', 'rb') as f:
            count_vect = pickle.load(f)
        #Load the saved model
        #print("Loading the model...")
        loaded_model = None
        with open('models/emotion_classifier.pk', 'rb') as f2:
            loaded_model = pickle.load(f2)

        #print("The model has been loaded...doing predictions now...")
        predictions = []
        sequence = [] #better formatting. 
        print("Text", textToAnalyze)
        for sentence in textToAnalyze:
            if sentence != " ":
                predictionIs = loaded_model.predict((count_vect.transform([sentence])))
                predictions.append(predictionIs)
                sequence.append(predictionIs[0])

        ############################################################################
        #JERRY'S CODE
        ############################################################################
        def read():
            # Call the Calendar API
            now = datetime.datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time
            print('Getting the upcoming 10 events')
            events_result = service.events().list(calendarId='primary', timeMin=now,
                                                  maxResults=10, singleEvents=True,
                                                  orderBy='startTime').execute()
            events = events_result.get('items', [])

            if not events:
                print('No upcoming events found.')
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                print(start, event['summary'])
            return events


        def order(t):
            work = dict({"assignment":"1", "project":"1", "work":"1", "homework":"1", "lab":"1", "report":"1", "paper":"1", "math":"1", "engineering":"1", "biology":"1", "physics":"1", "boring":"1", "job":"1", "computer":"1", "science":"1", "journal":"1", "lecture":"1", "tutorial":"1", "exam":"1", "assessment":"1", "test":"1"}) 
            costs = np.zeros(7)
            #either work or other
            now = datetime.datetime.utcnow()
            counter = 0
            current_day = now.replace(hour=0, minute=0, second=0,microsecond=0) + relativedelta(days=1)
            current_day_limit = current_day + relativedelta(days=1)
            while(counter < 7):
                events_result = service.events().list(calendarId='primary', timeMin=(current_day.isoformat()+'Z'),
                                                  timeMax=(current_day_limit.isoformat()+'Z'),
                                                  singleEvents=True,
                                                  orderBy='startTime').execute()
                events = events_result.get('items', [])
                for event in events:
                    description = ''
                    if(event.get('summary')):
                        description += event.get('summary') + ' : '
                    if(event.get('description')):
                        description += event.get('description')
                    is_work = False
                    for word in description.split():
                        if word in work.keys():
                            is_work = True
                            break
                    if((is_work and t or'work') or (not is_work and t!='work')):
                        costs[counter] += 1
                current_day = current_day_limit
                current_day_limit = current_day_limit + relativedelta(days=1)
                counter+=1

            lists = [1]
            for i in range(1, 7):
                counter = 0
                while(costs[i] > costs[counter]):
                    counter+=1
                lists.insert(counter, i+1)
            #for i in range(0, 7):
                #print(lists[i])
            return lists


        def analyze(order, duration):
            options = ['', '', '']
            number_options = 0
            for day in order:
                current_day = datetime.datetime.utcnow() + relativedelta(days=day)
                current_day = current_day.replace(hour=9, minute=0, second=0,microsecond=0)
                #current_day_limit = current_day + relativedelta(days=1)
                current_day_limit = current_day.replace(hour=23, minute=0, second=0, microsecond=0) - relativedelta(minutes=duration)
                current_time = current_day
                events_result = service.events().list(calendarId='primary', timeMin=current_day.isoformat() + 'Z',
                                                      timeMax=current_day_limit.isoformat() + 'Z',
                                                      singleEvents=True,
                                                      orderBy='startTime').execute()
                events = events_result.get('items', [])
                if not events:
                    #print('No upcoming events found.')
                    return current_time
                while(current_time <= current_day_limit):
                    #start = event['start'].get('dateTime')
                    #start = datetime.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S-04:00')
                    #end = event['end'].get('dateTime')
                    #end = datetime.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S-04:00')       
                    #print(datetime.datetime.now())
                    #print(datetime.datetime.now().replace(hour=23, microsecond=0).isoformat())
                    #test = relativedelta(minutes=15)
                    #start1 = end + relativedelta(minutes=15)
                    #end1 = start1 + relativedelta(minutes=duration+15)
                    #end2 = start - relativedelta(minutes=15)
                    #start2 = end2 - relativedelta(minutes=duration+15)


                    start = current_time - relativedelta(minutes=15)
                    end = current_time + relativedelta(minutes=duration+15)
                    #start = current_time
                    #end = current_time + relativedelta(minutes=duration)

                    body = {
                        "timeMin": start.isoformat()+'-04:00',
                        "timeMax": end.isoformat()+'-04:00',
                        "timeZone": 'America/New_York',
                        "items": [{"id": '3jerryliu@gmail.com'}]
                    }
                    eventsResult = service.freebusy().query(body=body).execute()
                    #print('The event result is: ')
                    #print(start)
                    #print(end)
                    #print(eventsResult)
                    #print(eventsResult[u'calendars'])
                    calendar_state = eventsResult[u'calendars']
                    #print(test)
                    email_state = calendar_state[u'3jerryliu@gmail.com']
                    #print(test)
                    busy_state = email_state[u'busy']
                    #print(test)
                    #print('end')
                    if(not busy_state):
                        options[number_options] = current_time.strftime("%Y-%m-%d %H:%M")
                        number_options+=1
                        current_time = current_time.replace(hour=23)
                        if(number_options==3):
                            return options
                    #cal_dict = eventsResult[u'calendars']
                    #for cal_name in cal_dict:
                    #    print(cal_name, cal_dict[cal_name])
                    current_time = current_time + relativedelta(minutes=15)
            return options

        def insert(name, duration, t):
            day_order = order(t)
            suggestions = analyze(day_order, duration)
            #for s in range(0, len(suggestions)):
            #   print(suggestions[s])
            #json_dump = json.dumps(suggestions, default=json_serial)
            #json_dump=json.dumps(suggestions, indent=4, sort_keys=True, default=str)
            #json_dump=json.dumps({"0": suggestions[0], "1": suggestions[1], "2": suggestions[2]}, sort_keys=True)
            suggestions ={"0": [suggestions[0]], "1": [suggestions[1]], "2": [suggestions[2]]}
            # #print(json_dump)
            # print(json_dump)
            #return json_dump
            return suggestions

        def schedule(name, duration, t, suggestion):
            #edit the parsing method below based on what the result of suggestion is expected to be
            suggestion = datetime.datetime.strptime(suggestion, '%Y-%m-%dT%H:%M:%S-04:00')
            #ask front end to pick which time they want
            answer = input('Would you like to have an event put on your calendar called ' + name + ' on ' + suggestion.strftime("%Y-%m-%d at %H:%M") + ' oclock for ' + str(duration) + ' minutes? ')
            if(answer != 'no'):
                suggestion_end = suggestion + relativedelta(minutes=duration)
                event = {
                  'summary': name,
                  'description': t,
                  'start': {
                    'dateTime': suggestion.isoformat()+ '-04:00',
                    'timeZone': 'America/New_York',
                  },
                  'end': {
                    'dateTime': suggestion_end.isoformat()+'-04:00',
                    'timeZone': 'America/New_York',
                  },
                  'reminders': {
                    'useDefault': False,
                    'overrides': [
                      {'method': 'email', 'minutes': 24 * 60},
                      {'method': 'popup', 'minutes': 10},
                    ],
                  },
                }

                event = service.events().insert(calendarId='primary', body=event).execute()
                print ('Event created: %s' % (event.get('htmlLink')))

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

        def use_neural_network(input_data):
            prediction = neural_network_model(x)
            with open('models/lexicon.pickle','rb') as f:
                lexicon = pickle.load(f)
                
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess,"models/model.ckpt")
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
                test = test[0]
                difference = abs(test[0] - test[1])
                if(difference >= 50):
                    result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
                    if result[0] == 0:
                        #print('Positive:',input_data)
                        return 0
                    elif result[0] == 1:
                        #print('Negative:',input_data)
                        return 1
                return 0.5


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
        saver = tf.train.import_meta_graph('models/model.ckpt.meta')
        

        sentiment = use_neural_network(textToAnalyzeWhole)

        ##########################################################3

        '''
        Find key action words: 
        '''
        powerWords = {'frustrated': 'Relax',
         'frustrating': 'Relax',
         'chill': 'Relax',
         'me': 'Depressed',
         'much': 'Relax',
         'my': 'Relax',
         'not': 'Relax',
         'overwhelmed': 'Relax',
         'vacation': 'Relax',
         'crazy': 'Relax',
         'stress': 'Relax',
         'stressed': 'Relax',
         'too': 'Relax',
         'sleep': 'Depressed',
         'burnt': 'Relax',
         'food': 'Relax',
         'control': 'Relax',
         'work': 'Action2',
         'relax': 'Relax',
         'relaxation': 'Relax',
         'hesistate': 'Procrast',
         'lazy': 'Procrast',
         'prolong': 'Procrast',
         'slow': 'Procrast',
         'apathetic': 'Procrast',
         'bored': 'Procrast',
         'boring': 'Procrast',
         'tedium': 'Procrast',
         'anime': 'Procrast',
         'netflix': 'Procrast',
         'movies': 'Procrast',
         'waste': 'Procrast',
         'ice cream': 'Procrast',
         'snack': 'Procrast',
         'binge': 'Procrast',
         'tv': 'Procrast',
         'game': 'Procrast',
         'video': 'Procrast',
         'facebook': 'Procrast',
         'twitter': 'Procrast',
         'instagram': 'Procrast',
         'twitch': 'Procrast',
         'league': 'Procrast',
         'guilt': 'Procrast',
         'shame': 'Procrast',
         'procrastinate': 'Procrast',
         'procrastination': 'Procrast',
         'procrastinated': 'Procrast',
         'wasted': 'Procrast',
         'time': 'Procrast',
         'hesitated': 'Procrast',
         'prolonged': 'Procrast',
         'procrastinating': 'Procrast',
         'wasting': 'Procrast',
         'dark': 'Depressed',
         'destroy': 'Depressed',
         'die': 'Depressed',
         'hate': 'Depressed',
         'kill': 'Depressed',
         'life': 'Depressed',
         'murder': 'Depressed',
         'myself': 'Depressed',
         'revenge': 'Depressed',
         'someone': 'Depressed',
         'understand': 'Depressed',
         'cry': 'Depressed',
         'worst': 'Depressed',
         'enemy': 'Depressed',
         'hurts': 'Depressed',
         'broken': 'Depressed',
         'erase': 'Depressed',
         'evil': 'Depressed',
         'pain': 'Depressed',
         'emotion': 'Depressed',
         'world': 'Depressed',
         'numb': 'Depressed',
         'emotions': 'Depressed',
         'supposed': 'Depressed',
         'strength': 'Depressed',
         'alone': 'Depressed',
         'depression': 'Depressed',
         'depressed': 'Depressed',
         'suicide': 'Depressed',
         'tears': 'Depressed',
         'tear': 'Depressed',
         'hole': 'Depressed',
         'chasm': 'Depressed',
         'burden': 'Depressed',
         'sad': 'Depressed',
         'died': 'Depressed',
         'cried': 'Depressed',
         'asleep': 'Depressed',
         'haze': 'Depressed',
         'energy': 'Depressed',
         'dreading': 'Depressed',
         'do': 'Action2',
         'find': 'Action1',
         'go': 'Action2',
         'need': 'Action1',
         'have': 'Action1',
         'play': 'Action1',
         'want': 'Action1',
         'must': 'Action1',
         'require': 'Action1',
         'required': 'Action1',
         'wish': 'Action1',
         'crave': 'Action1',
         'miss': 'Action1',
         'accomplish': 'Action2',
         'finish': 'Action2',
         'ace': 'Action2',
         'pass': 'Action2',
         'earn': 'Action2',
         'build': 'Action2',
         'achieve': 'Action2',
         'win': 'Action2',
         'create': 'Action2',
         'implement': 'Action2',
         'perform': 'Action2',
         'soon': 'Action2',
         'possible': 'Action2',
         'buy': 'Action2',
         'complete': 'Action2',
         'start': 'Action2',
         'exercise': 'Action2',
         'visit': 'Action2',
         'more': 'Action2',
         'use': 'Action2',
         'make': 'Action2',
         'try': 'Action2',
         'study': 'Action2',
         'accomplishing': 'Action2',
         'finishing': 'Action2',
         'aceing': 'Action2',
         'passing': 'Action2',
         'earning': 'Action2',
         'building': 'Action2',
         'achieveing': 'Action2',
         'wining': 'Action2',
         'createing': 'Action2',
         'implementing': 'Action2',
         'doing': 'Action2',
         'performing': 'Action2',
         'sooning': 'Action2',
         'possibleing': 'Action2',
         'buying': 'Action2',
         'completeing': 'Action2',
         'starting': 'Action2',
         'working': 'Action2',
         'exerciseing': 'Action2',
         'visiting': 'Action2',
         'moreing': 'Action2',
         'useing': 'Action2',
         'makeing': 'Action2',
         'trying': 'Action2',
         'studying': 'Action2',
         'going': 'Action2',
         'learn': 'Action2'}
        workWords = dict({"assignment":"1", "project":"1", "work":"1", "homework":"1", "lab":"1", "report":"1", "paper":"1", "math":"1", "engineering":"1", "biology":"1", "physics":"1", "boring":"1", "job":"1", "computer":"1", "science":"1", "journal":"1", "lecture":"1", "tutorial":"1", "exam":"1", "assessment":"1", "test":"1"}) 
            
        procrastinateCount = 0
        depressionCount = 0
        needRelaxCount = 0
        actionToggled = False
        actionSentence = ''
        sentenceAdded = False
        isWork = False
        workState = False
        i = 0
        for sentence in textToAnalyze:
            sentenceAdded = False
            if(sentence != " "):
                words = re.sub("[^\w]", " ",  sentence).split()
                #See if words are in the hashmap
                for word in words:
                    if word in workWords:
                        isWork = True
                    if word in powerWords:
                        if powerWords[word] == 'Depressed' and predictions[i][0] != "joy":
                            depressionCount += 1
                        if powerWords[word] == 'Relax' and predictions[i][0] != "joy":
                            needRelaxCount += 1
                        if powerWords[word] == 'Procrast' and predictions[i][0] != "joy":
                            procrastinateCount += 1
                        if powerWords[word] == 'Action1' or powerWords[word] == 'Action2':
                            if actionToggled == True:
                                if sentenceAdded is False:
                                    actionSentence = sentence
                                    sentenceAdded = True
                                    workState = isWork
                            else:
                                actionToggled = True

            if i < len(predictions):
                i += 1

        print(workState)
        #Evaluate
        pain = max(depressionCount, needRelaxCount, procrastinateCount)
        if pain > int(len(textToAnalyze)*0.4):
            if procrastinateCount >= pain:
                state = "procrastinate"
            if needRelaxCount >= pain:
                state = "relax"
            if depressionCount >= pain:
                state = "depression"
        else:
            state = "neutral"

        #Apart from primary algo, can miss words.
        counter = 0 
        for label in sequence:
            if label == "sadness":
                counter += 1

        if counter >= int(len(textToAnalyze)*0.6) and counter >= 3:
            state = "depression"

        print(procrastinateCount)
        print(needRelaxCount)
        print(depressionCount)
        print(state)

        SCOPES = 'https://www.googleapis.com/auth/calendar'
        store = file.Storage('credentials.json')
        creds = store.get()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
            creds = tools.run_flow(flow, store)
        service = discovery.build('calendar', 'v3', http=creds.authorize(Http()))
        #j = 0

        if(actionSentence != ""):
            suggestions = insert(actionSentence, 60, 'work')
                # for sentence in actionSentences:
                #     if str(j) in suggestions:
                #         suggestions[str(j)].append(sentence)
                #     j += 1

        else: 
            suggestions = {}

        # prediction_series = list(pd.Series(predictions))

        # final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))

        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        #print(predictions)

        #Create dictionary with everything that I need to return. 

        responses = jsonify(predictions=sequence, sentence = actionSentence, mindState=state, calSuggestions=suggestions, sentimentState = sentiment)
        responses.status_code = 200

        return (responses)

@app.route('/calendar', methods=['POST'])

def updateCalender():
    textToAnalyze_json = request.get_json()
    textToAnalyze = json.loads(json.dumps([textToAnalyze_json['selection'],textToAnalyze_json['sentence'],textToAnalyze_json['minutes']]))
    # print(textToAnalyze['selection'])
    # print(textToAnalyze['sentence'])
    # print(textToAnalyze['minutes'])
    print(textToAnalyze)

    def read():
        # Call the Calendar API
        now = datetime.datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time
        print('Getting the upcoming 10 events')
        events_result = service.events().list(calendarId='primary', timeMin=now,
                                              maxResults=10, singleEvents=True,
                                              orderBy='startTime').execute()
        events = events_result.get('items', [])

        if not events:
            print('No upcoming events found.')
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            print(start, event['summary'])
        return events


    def order(t):
        work = dict({"assignment":"1", "project":"1", "work":"1", "homework":"1", "lab":"1", "report":"1", "paper":"1", "math":"1", "engineering":"1", "biology":"1", "physics":"1", "boring":"1", "job":"1", "computer":"1", "science":"1", "journal":"1", "lecture":"1", "tutorial":"1", "exam":"1", "assessment":"1", "test":"1"}) 
        costs = np.zeros(7)
        #either work or other
        now = datetime.datetime.utcnow()
        counter = 0
        current_day = now.replace(hour=0, minute=0, second=0,microsecond=0) + relativedelta(days=1)
        current_day_limit = current_day + relativedelta(days=1)
        while(counter < 7):
            events_result = service.events().list(calendarId='primary', timeMin=(current_day.isoformat()+'Z'),
                                              timeMax=(current_day_limit.isoformat()+'Z'),
                                              singleEvents=True,
                                              orderBy='startTime').execute()
            events = events_result.get('items', [])
            for event in events:
                description = ''
                if(event.get('summary')):
                    description += event.get('summary') + ' : '
                if(event.get('description')):
                    description += event.get('description')
                is_work = False
                for word in description.split():
                    if word in work.keys():
                        is_work = True
                        break
                if((is_work and t or'work') or (not is_work and t!='work')):
                    costs[counter] += 1
            current_day = current_day_limit
            current_day_limit = current_day_limit + relativedelta(days=1)
            counter+=1

        lists = [1]
        for i in range(1, 7):
            counter = 0
            while(costs[i] > costs[counter]):
                counter+=1
            lists.insert(counter, i+1)
        #for i in range(0, 7):
            #print(lists[i])
        return lists


    def analyze(order, duration):
        options = ['', '', '']
        number_options = 0
        for day in order:
            current_day = datetime.datetime.utcnow() + relativedelta(days=day)
            current_day = current_day.replace(hour=9, minute=0, second=0,microsecond=0)
            #current_day_limit = current_day + relativedelta(days=1)
            current_day_limit = current_day.replace(hour=23, minute=0, second=0, microsecond=0) - relativedelta(minutes=duration)
            current_time = current_day
            events_result = service.events().list(calendarId='primary', timeMin=current_day.isoformat() + 'Z',
                                                  timeMax=current_day_limit.isoformat() + 'Z',
                                                  singleEvents=True,
                                                  orderBy='startTime').execute()
            events = events_result.get('items', [])
            if not events:
                #print('No upcoming events found.')
                return current_time
            while(current_time <= current_day_limit):
                #start = event['start'].get('dateTime')
                #start = datetime.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S-04:00')
                #end = event['end'].get('dateTime')
                #end = datetime.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S-04:00')       
                #print(datetime.datetime.now())
                #print(datetime.datetime.now().replace(hour=23, microsecond=0).isoformat())
                #test = relativedelta(minutes=15)
                #start1 = end + relativedelta(minutes=15)
                #end1 = start1 + relativedelta(minutes=duration+15)
                #end2 = start - relativedelta(minutes=15)
                #start2 = end2 - relativedelta(minutes=duration+15)


                start = current_time - relativedelta(minutes=15)
                end = current_time + relativedelta(minutes=duration+15)
                #start = current_time
                #end = current_time + relativedelta(minutes=duration)

                body = {
                    "timeMin": start.isoformat()+'-04:00',
                    "timeMax": end.isoformat()+'-04:00',
                    "timeZone": 'America/New_York',
                    "items": [{"id": '3jerryliu@gmail.com'}]
                }
                eventsResult = service.freebusy().query(body=body).execute()
                #print('The event result is: ')
                #print(start)
                #print(end)
                #print(eventsResult)
                #print(eventsResult[u'calendars'])
                calendar_state = eventsResult[u'calendars']
                #print(test)
                email_state = calendar_state[u'3jerryliu@gmail.com']
                #print(test)
                busy_state = email_state[u'busy']
                #print(test)
                #print('end')
                if(not busy_state):
                    options[number_options] = current_time.strftime("%Y-%m-%d %H:%M")
                    number_options+=1
                    current_time = current_time.replace(hour=23)
                    if(number_options==3):
                        return options
                #cal_dict = eventsResult[u'calendars']
                #for cal_name in cal_dict:
                #    print(cal_name, cal_dict[cal_name])
                current_time = current_time + relativedelta(minutes=15)
        return options

    def insert(name, duration, t):
        day_order = order(t)
        suggestions = analyze(day_order, duration)
        #for s in range(0, len(suggestions)):
        #   print(suggestions[s])
        #json_dump = json.dumps(suggestions, default=json_serial)
        #json_dump=json.dumps(suggestions, indent=4, sort_keys=True, default=str)
        json_dump=json.dumps({"0": suggestions[0], "1": suggestions[1], "2": suggestions[2]}, sort_keys=True)
        #print(json_dump)
        print(json_dump)
        return json_dump

    def schedule(name, duration, t, suggestion):
        #edit the parsing method below based on what the result of suggestion is expected to be
        suggestion = datetime.datetime.strptime(suggestion, '%Y-%m-%d %H:%M')
        #ask front end to pick which time they want
        suggestion_end = suggestion + relativedelta(minutes=duration)
        event = {
          'summary': name,
          'description': t,
          'start': {
            'dateTime': suggestion.isoformat()+ '-04:00',
            'timeZone': 'America/New_York',
          },
          'end': {
            'dateTime': suggestion_end.isoformat()+'-04:00',
            'timeZone': 'America/New_York',
          },
          'reminders': {
            'useDefault': False,
            'overrides': [
              {'method': 'email', 'minutes': 24 * 60},
              {'method': 'popup', 'minutes': 10},
            ],
          },
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        print ('Event created: %s' % (event.get('htmlLink')))

    # Setup the Calendar API
    SCOPES = 'https://www.googleapis.com/auth/calendar'
    store = file.Storage('credentials.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = discovery.build('calendar', 'v3', http=creds.authorize(Http()))
    work = dict({"assignment":"1", "project":"1", "work":"1", "homework":"1", "lab":"1", "report":"1", "paper":"1", "math":"1", "engineering":"1", "biology":"1", "physics":"1", "boring":"1", "job":"1", "computer":"1", "science":"1", "journal":"1", "lecture":"1", "tutorial":"1", "exam":"1", "assessment":"1", "test":"1"}) 
    isWork = False
    sentence = textToAnalyze[1]
    for word in sentence:
        if word in work:
            isWork = True
    if isWork == True:
        schedule(sentence, int(textToAnalyze[2]), 'work', textToAnalyze[0])
    else:
        schedule(sentence, int(textToAnalyze[2]), 'other', textToAnalyze[0])

    #See if it's in work. 

    responses = jsonify(response=['Done!'])
    responses.status_code = 200
    return(responses)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
