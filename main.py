import sys
import numpy as np
import cv2
import glfw
from mqoloader.loadmqo import LoadMQO
import Application

width  = 640
height = 480
app = Application.Application('Hand marker AR', width, height, 0, 0)

model_filename = "./mqo/ninja.mqo"
model_scale = 1.0
model = LoadMQO(model_filename, model_scale, 0)
app.set_mqo_model(model)

while not app.glwindow.window_should_close():
    app.display_func(app.glwindow.window)    
    glfw.poll_events()

glfw.terminate()


