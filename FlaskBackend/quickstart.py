"""
Shows basic usage of the Google Calendar API. Creates a Google Calendar API
service object and outputs a list of the next 10 events on the user's calendar.
"""
from __future__ import print_function
from apiclient import discovery
from httplib2 import Http
from oauth2client import file, client, tools
import datetime

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

def insert(name, length, t):
	events = read()

	for event in events:
	    start = event['start'].get('dateTime', event['start'].get('date'))
	    print(start, event['summary'])

	event = {
	  'summary': name,
	  'description': t,
	  'start': {
	    'dateTime': '2018-05-28T09:00:00-07:00',
	    'timeZone': 'America/Los_Angeles',
	  },
	  'end': {
	    'dateTime': '2018-05-28T17:00:00-07:00',
	    'timeZone': 'America/Los_Angeles',
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