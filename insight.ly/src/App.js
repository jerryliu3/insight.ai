import React, { Component } from 'react';
import './App.css';
import Main from './main.js';
import Amplify from 'aws-amplify';
import aws_exports from './aws-exports';
import { Analytics } from 'aws-amplify';

Amplify.configure(aws_exports);

class App extends Component {
  render() {
  	Analytics.record('appRender');
    return (
      <div>
        <Main/>
      </div>
    );
  }
}

export default App;
