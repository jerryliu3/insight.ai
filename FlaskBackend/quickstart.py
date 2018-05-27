"""
Shows basic usage of the Google Calendar API. Creates a Google Calendar API
service object and outputs a list of the next 10 events on the user's calendar.
"""
from __future__ import print_function
from apiclient import discovery
from httplib2 import Http
from oauth2client import file, client, tools
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import json
# Setup the Calendar API
SCOPES = 'https://www.googleapis.com/auth/calendar'
store = file.Storage('credentials.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
    creds = tools.run_flow(flow, store)
service = discovery.build('calendar', 'v3', http=creds.authorize(Http()))

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
	#	print(suggestions[s])
	#json_dump = json.dumps(suggestions, default=json_serial)
	#json_dump=json.dumps(suggestions, indent=4, sort_keys=True, default=str)
	json_dump=json.dumps({"0": suggestions[0], "1": suggestions[1], "2": suggestions[2]}, sort_keys=True)
	#print(json_dump)
	print(json_dump)
	return json_dump

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