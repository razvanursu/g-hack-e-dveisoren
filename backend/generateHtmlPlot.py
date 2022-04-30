import plotly.express as px
import os
os.chdir("/Users/pauldelseith/Documents/TUM/Master/2. Semester/GHack/Code/g-hack-e-dveisoren/backend")
fig =px.scatter(x=range(10), y=range(10))
fig.write_html("templates/energyGen/energyPlot.html")