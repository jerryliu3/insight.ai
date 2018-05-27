import React, {Component} from 'react';
import {Line} from 'react-chartjs-2';

class Chart extends Component{
      constructor(props ){
        super(props);
        this.state={
          chartData:{
            labels:['Joy', 'Fear', 'Anger', 'Disgust', 'Sadness'],
            dataSets:[{
              label: 'Proportion',
              data:[1, 2, 3, 4, 5],
              backgroundColor:[
                'rgba(255, 99, 132, 0.6)',
                'rgba(54, 162, 235, 0.6)',
                'rgba(255, 206, 86, 0.6)',
                'rgba(75, 192, 192, 0.6)',
                'rgba(153, 102, 255, 0.6)'
              ],
            }]
          }
        }
    }
	render(){
		return(
			<div className="chart">
			<Line
				data={this.state.chartData}
				options={{
					maintainAspectRatio:false
				}}

			/>
			</div>
		)
	}
}

export default Chart;