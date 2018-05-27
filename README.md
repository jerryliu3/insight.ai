DEMO: http://insightly-hosting-mobilehub-330001789.s3-website.ap-southeast-1.amazonaws.com/

## Introduction

This is Insight.ai, a text-based natural language processing (NLP) web application that provides qualitative and quantitative information on user writing such as mood, tasks and sources of stress. Built for RUHacks 2018, this project is targeted towards those who use a diary or writing as a form of reflection, as well as anyone who wants to wants to extract more meaning out of their/someone else's writing (emails, love letters, etc).

## Requirements

This module requires the following modules/libraries:

* React
* Python
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

The main purpose of the application is to provide insight on personal writing, which will usually come from the users themselves.

Built with a React front-end and hosted on AWS with a Python back-end.

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

* Andrew Luo
* [Arri Ye](https://github.com/music-mind/)
* [Jerry Liu](https://github.com/jerryliu3)




