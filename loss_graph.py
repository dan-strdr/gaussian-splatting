import plotly.express as px
import pandas as pd
import numpy as np

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

log_path = 'logs/training-1211630.out'

with open(log_path) as log_file:
    data_lines = log_file.readlines()


loss_values = []

for line in data_lines:
    if 'Loss=' in line:
        loss_values.append(float(line[line.find('=')+1: line.find(']')]))

loss_values = np.array(loss_values)
loss_values = smooth(loss_values, 20)[:-20]
iterations = np.linspace(0, 30000, len(loss_values))

df = pd.DataFrame(data={'Iteration': iterations, 'Loss': loss_values})
fig = px.line(df, x="Iteration", y="Loss", title='Combined Loss Value over Optimization')
fig.write_image("loss_curve.jpg")