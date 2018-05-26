import React, { Component } from 'react';
import TextField from '@material-ui/core/TextField';
import { withStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';

const styles = theme => ({
  textField: {
    marginLeft: theme.spacing.unit,
    marginRight: theme.spacing.unit,
    width: 500,
  },
  button: {
    margin: theme.spacing.unit,
  },
  input: {
    display: 'none',
  },
});

class Main extends Component {
  render() {
    const { classes } = this.props;

    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">Insight.ly</h1>
        </header>
        <TextField
          id="Input Field"
          label="Analyze some text!"
          margin="normal"
          className={classes.textField}
          multiline
        />
        <Button variant="raised" color="primary" className={classes.button}>
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
      </div>
    );
  }
}

export default withStyles(styles)(Main);