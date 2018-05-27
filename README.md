DEMO: http://insightly-hosting-mobilehub-330001789.s3-website.ap-southeast-1.amazonaws.com/

## Introduction

This is Insight.ai, a text-based natural language processing (NLP) web application that provides qualitative and quantitative information on user writing such as mood, tasks and sources of stress. After analyzing the information, it will provide personalized feedback to the user and suggest courses of action or calendar task/event planning. Built for RUHacks 2018, this project is targeted towards those who use a diary or writing as a form of reflection, as well as anyone who wants to wants to extract more meaning out of their/someone else's writing (emails, love letters, etc).

## Requirements

Built with a React front-end and hosted on AWS with a Python back-end.

This module requires the following modules/libraries:

* React
* Python (only 3.6 tested)
* Flask
* AWS
* TensorFlow
* Scikit-learn
* NLTK
* Google Calendar API
* Data for model training: http://help.sentiment140.com/for-students/
and more...

## Installation

``git clone
npm install``

## Purpose

The main purpose of the application is to provide insight on personal writing, for diary-users, writers, and anyone looking for feedback. Diaries have been shown to have numerous health benefits [1-3], and in an age where mental health is on the rise and personal information is exploding, the ability to keep track and reflect on oneself has never been greater. Even after taking the time to write, it can be difficult for many people to extract significant meaning from it. Furthermore, people may not take what they write to heart, thinking they know themselves well enough, which has been shown might not be the case [4]. 

Past solutions to this problem were primitive, only providing a structure for someone to type and store notes. The recent growth in machine learning and natural language processing has allowed for new solutions in retrospection and estimating sentiment or more personalized task management for future planning. However, no solutions exist that bridge the gap between these two to provide users with a new look on what they've done in the past and suggestions on how to organize themselves for the future.

Insight.ai does this through a friendly user interface where users can input an arbitrary amount of sentences and save it to a database to read later. At the click of another button, the system will estimate the person's overall mood and per sentence using cross-validation between a neural network and naive bayes classifier. Based on the estimate and the topic of the user, suggestions will be made on what the user may want to do. For example, if the user is very stressed about homework, the system will encourage them to take more breaks and relax. Additionally, if the user talks about not doing something or wanting to do something, the system will automatically interpret it and suggest up to 3 timeslots in their next week calendar to do it. The timeslots always open and spread out on their current calendar, and are calculated using a priority-based algorithm based on the category of the event. For example, if it is a homework task, the system will not suggest it on days that already have a high relative amount of work.

Other applications for this technology, as mentioned, include analyzing emails to determine an appropriate response, or decoding the feelings behind someone's messages to you.

## Problems We Encountered

* Lots of datasets for sentiment based off of tweets or messages but not on personal writing or essays
* TensorFlow is not the best library for natural language processing
* Rewriting code for back-end server deployment gave lots of unexpected errors
* Time formatting inconsistances while using Google Calendar API

## Future Todos

* Keyword extraction of text using NLP:
* https://www.airpair.com/nlp/keyword-extraction-tutorial
* https://www.quora.com/What-are-the-best-keyword-extraction-algorithms-for-natural-language-processing-and-how-can-they-be-implemented-in-Python
* https://github.com/csurfer/rake-nltk

## Maintainers

* [Andrew Luo](https://github.com/Andrew-Luo1)
* [Arri Ye](https://github.com/music-mind/)
* [Jerry Liu](https://github.com/jerryliu3)

## References

[1] http://www.apa.org/monitor/sep01/keepdiary.aspx
[2] https://www.huffingtonpost.ca/2017/06/20/benefits-of-journaling_n_17212154.html
[3] https://onlinelibrary.wiley.com/doi/full/10.1002/jclp.22463
[4] https://www.sciencedaily.com/releases/2010/02/100226093235.htm
