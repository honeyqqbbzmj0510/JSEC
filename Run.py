import sys
import numpy as np
from datetime import datetime

Run_Time = datetime.today().date()

class Logger(object):
    def __init__(self, filename='Run_recording_{}'.format(Run_Time), stream=sys.stdout):
        self.terminal = stream
        self.version = int(np.loadtxt('./version.txt', delimiter=',')[0])
        self.log = open(filename + str(self.version) + '.log', 'a+')
        self.run = 1

    def write(self, message):
        message = message
        self.terminal.write(message)
        self.log.write(message)

    def show_version(self):
        print("_____________________Version=={}_____________________________".format(self.version))
        np.savetxt('./version.txt', [self.version + self.run, self.version], delimiter=',')

    def flush(self):
        pass