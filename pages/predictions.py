import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import shap
import pylint


from joblib import load
model = load('assets/model.joblib')
print('Model Loaded Successfully')

def predict(Year, Horsepower, Doors, Transmission):
  new_df = pd.DataFrame(
      data=[[Year, Horsepower, Doors, Transmission]],
      columns=['Year', 'Horsepower', 'Doors', 'Transmission']
  )
  pred = model.predict(new_df)[0]

  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(new_df)

  feature_names = new_df.columns
  feature_values = new_df.values[0]
  shaps = pd.Series(shap_values[0], zip(feature_names, feature_values))

  result = [html.Div(f'MSRP is Estimated at ${pred:,.0f} for this Vehicle. \n\n')]
  result.append(html.Div(f'Starting from a baseline of ${explainer.expected_value:,.0f}. \n'))
  explanation = shaps.to_string()
  lines = explanation.split('\n')
  for line in lines:
      result.append(html.Div(line))
  return result

result = predict(Year=2011, Horsepower=335, Doors=2, Transmission=1)

column1 = dbc.Col(
    [
        dcc.Markdown('#### Year'),
        dcc.Dropdown(
    options=[
        {'label': '1990', 'value': '90'},
        {'label': '1991', 'value': '91'},
        {'label': '1992', 'value': '92'},
        {'label': '1993', 'value': '93'},
        {'label': '1994', 'value': '94'},
        {'label': '1995', 'value': '95'},
        {'label': '1996', 'value': '96'},
        {'label': '1997', 'value': '97'},
        {'label': '1998', 'value': '98'},
        {'label': '1999', 'value': '99'},
        {'label': '2000', 'value': '00'},
        {'label': '2001', 'value': '01'},
        {'label': '2002', 'value': '02'},
        {'label': '2003', 'value': '03'},
        {'label': '2004', 'value': '04'},
        {'label': '2005', 'value': '05'},
        {'label': '2006', 'value': '06'},
        {'label': '2007', 'value': '07'},
        {'label': '2008', 'value': '08'},
        {'label': '2009', 'value': '09'},
        {'label': '2010', 'value': '10'},
        {'label': '2011', 'value': '11'},
        {'label': '2012', 'value': '12'},
        {'label': '2013', 'value': '13'},
        {'label': '2014', 'value': '14'},
        {'label': '2015', 'value': '15'},
        {'label': '2016', 'value': '16'},
        {'label': '2017', 'value': '17'},
    ],
    placeholder="Select a Year",
), 
dcc.Markdown('#### Horsepower'), 
dcc.Input(
        id='horsepower', 
        placeholder='Enter Horsepower...',
        type='text',
        value=''
), 
dcc.Markdown('#### Doors'), 
dcc.Slider(
    id='doors', 
    min=2, 
    max=4, 
    step=1, 
    value=2, 
    marks={n: str(n) for n in range(2,5,1)}, 
    className='mb-5', 
 ),
 dcc.Markdown('#### Transmission Type'),
 dcc.Dropdown(
     id='transmission',
    options=[
        {'label': 'Manual', 'value': 'M'},
        {'label': 'Automatic', 'value': 'A'},
    ],
    value='MTL'
)   
    ],
    md=4,
)

column2 = dbc.Col(
    [
        html.H2('Estimated MSRP for This Vehicle', className='mb-5'), 
        html.Div(result, id='prediction-content', className='lead')
    ]
)

layout = dbc.Row([column1, column2])