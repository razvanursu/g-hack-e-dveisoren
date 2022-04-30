import numpy as np
import plotly.express as px
import os

def createPredictionPlot():
    os.chdir("/Users/pauldelseith/Documents/TUM/Master/2. Semester/GHack/Code/g-hack-e-dveisoren/backend")
    fig =px.scatter(x=range(15), y=range(15))
    fig.write_html("templates/energyGen/energyPlot.html")
    return
