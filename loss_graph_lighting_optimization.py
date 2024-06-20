import plotly.express as px
import pandas as pd
import numpy as np

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

log_path = 'logs/training-1222187.out'

with open(log_path) as log_file:
    data_lines = log_file.readlines()


loss_values = []

for line in data_lines:
    if 'loss:' in line:
        loss_values.append(float(line[line.find(':')+1: line.find('[')]))

loss_values = np.array(loss_values)
loss_values = smooth(loss_values, 10)[10: -10]
iterations = np.linspace(0, 1000, len(loss_values))

df = pd.DataFrame(data={'Iteration': iterations, 'Loss': loss_values})
fig = px.line(df, x="Iteration", y="Loss", title='Lighting Optimization Loss Value over Optimization')
fig.write_image("loss_curve_lighting_optimization.jpg")

coordinate_values = []

for line in data_lines:
    if 'light_pos:' in line:
        coordinate_values.append(eval(line[line.find('(')+1: line.find('],')+1]))


coordinate_values_x = [each[0] for each in coordinate_values]
coordinate_values_y = [each[1] for each in coordinate_values]
coordinate_values_z = [each[2] for each in coordinate_values]

target_values_x = [0] * len(coordinate_values_x)
target_values_y = [1.5] * len(coordinate_values_y)
target_values_z = [0.1] * len(coordinate_values_z)

iterations = np.linspace(0, 1000, len(coordinate_values))

coordinate_values_x_df = pd.DataFrame(data={'iterations': iterations, 'values': coordinate_values_x, 'type': ['coordinate_values_x']*len(coordinate_values_x)})
coordinate_values_y_df = pd.DataFrame(data={'iterations': iterations, 'values': coordinate_values_y, 'type': ['coordinate_values_y']*len(coordinate_values_y)})
coordinate_values_z_df = pd.DataFrame(data={'iterations': iterations, 'values': coordinate_values_z, 'type': ['coordinate_values_z']*len(coordinate_values_z)})

target_values_x_df = pd.DataFrame(data={'iterations': iterations, 'values': target_values_x, 'type': ['target_values_x']*len(target_values_x)})
target_values_y_df = pd.DataFrame(data={'iterations': iterations, 'values': target_values_y, 'type': ['target_values_y']*len(target_values_y)})
target_values_z_df = pd.DataFrame(data={'iterations': iterations, 'values': target_values_z, 'type': ['target_values_z']*len(target_values_z)})

df = pd.concat([coordinate_values_x_df, coordinate_values_y_df, coordinate_values_z_df, target_values_x_df, target_values_y_df, target_values_z_df])

fig = px.line(df, x="iterations", y="values", color='type', title='Lighting Optimization Coordinate Values over Optimization')
fig.write_image("coordinates_lighting_optimization.jpg")