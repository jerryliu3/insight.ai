import React, { Component } from 'react';
import TextField from '@material-ui/core/TextField';
import { withStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import uuid from 'uuid/v4';
import AWS from 'aws-sdk';
import Paper from '@material-ui/core/Paper';
import Typography from '@material-ui/core/Typography';
import { VictoryPie, VictoryContainer } from 'victory';

// AWS.config.update({
//   credentials: new AWS.CognitoIdentityCredentials({
//     IdentityPoolId: 'us-east-1_eH5uzHwEY'
//   }),
//   region:'us-east-1',
// });

// const dynamodb = new AWS.DynamoDB.DocumentClient({
//   credentials: new AWS.CognitoIdentityCredentials({
//     IdentityPoolId: 'us-east-1_eH5uzHwEY'
//   }),
//   region:'us-east-1',
// }); 

const styles = theme => ({
  textField: {
    // marginLeft: theme.spacing.unit,
    // marginRight: theme.spacing.unit,
    textAlign: "left",
    float: "left",
    marginLeft: 30,
    width: 500,
  },
  button: {
    margin: theme.spacing.unit,
    verticalAlign: "top"
  },
  input: {
    display: 'none',
  },
  root: theme.mixins.gutters({
    paddingTop: 16,
    paddingBottom: 16,
    marginTop: theme.spacing.unit * 3,
  }),
});

class Main extends Component {

  state = {
    text: '',
    prediction: '',
    predictions: [],
  };

  saveText = e => {
    let { text } = this.state;
    let predictions = this.state.predictions.concat([]);
    let sentences = text.match( /[^\.!\?]+[\.!\?]+/g );
    if (!sentences) sentences = [text];
    sentences.forEach(sentence => {
      let str = JSON.stringify(sentence);
      fetch('http://184.72.144.143/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: str,
        })
      })
      .then(async res => {
        let result = await res.json();
        console.log(result);
        console.log(predictions.concat([result.predictions]));
        let wait = await this.setState({
          prediction: result.predictions,
          predictions: predictions.concat([result.predictions])
        });
        predictions = predictions.concat([result.predictions]);
      });
    });
    // let user_id = uuid();
    // let params = {
    //   Item: { /* required */
    //     'userId': user_id,
    //     'name': "John Doe",
    //     'entries': [
    //       text
    //     ]
    //       /* '<AttributeName>': ... */
    //     },
    //   TableName: 'Insights', /* required */
    //   };
    // dynamodb.put(params, function(err, data) {
    //   if (err) console.log(err, err.stack); // an error occurred
    //   else     console.log(data);           // successful response
    // });
  }

  updateText = e => {
    this.setState({
      text: e.target.value
    });
  }

  mode = array => {
    if(array.length == 0)
        return null;
    let modeMap = {};
    let maxEl = array[0];
    let maxCount = 1;
    for(let i = 0; i < array.length; i++)
    {
        let el = array[i];
        if(modeMap[el] == null)
            modeMap[el] = 1;
        else
            modeMap[el]++;  
        if(modeMap[el] > maxCount)
        {
            maxEl = el;
            maxCount = modeMap[el];
        }
    }
    return maxEl;
  }

  render() {
    const { classes } = this.props;
    let { text, prediction, predictions } = this.state;

    let overall_prediction = this.mode(predictions);

    let prediction_display = prediction ? <Paper className={classes.root} elevation={0}>
        <Typography variant="headline" component="h3">
          {overall_prediction}
        </Typography>
        <Typography component="p">
          Are you feeling this way?
        </Typography>
      </Paper> : "";

    let predictions_display = (predictions.length > 0) ? <Paper className={classes.root} elevation={4}>
        <Typography variant="headline" component="h3">
          {predictions.join(" ")}
        </Typography>
        <Typography component="p">
          Are you feeling this way?
        </Typography>
      </Paper> : "";

    let unique_predictions = predictions.filter((value, index, self) => self.indexOf(value) === index);
    let chart_data = [];
    unique_predictions.map(prediction => {
      const freq = predictions.filter(i => i === prediction).length;
      chart_data.push({x: prediction, y: freq});
    });

    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title"><a 
          href="https://github.com/jerryliu3/insight.ai" 
          target="_blank">Insight.ai</a></h1>
        </header>
        <span>
        <TextField
          id="Input Field"
          label="Analyze some text!"
          margin="normal"
          value={text}
          onChange={this.updateText}
          className={classes.textField}
          multiline
        />
        <p> </p>
        </span>
      <span className="buttons">
        <Button variant="raised" color="primary" className={classes.button} onClick={this.saveText}>
        Analyze
        </Button>
        <input
          accept="image/*"
          className={classes.input}
          id="raised-button-file"
          multiple
          type="file"
        />
        <label htmlFor="raised-button-file">
          <Button variant="raised" component="span" className={classes.button}>
            Upload
          </Button>
        </label>
        {(chart_data.length > 0 ? 
          <span className="chart">
          <VictoryPie
            colorScale={["tomato", "orange", "green", "cyan", "navy" ]}
            data={chart_data}
            containerComponent={<VictoryContainer responsive={false} width={420} />}
          /></span> : ""
        )}
        </span>
        {prediction_display}  
      </div>
    );
  }
}

export default withStyles(styles)(Main);