import datetime as dt

class Timer:
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()
        print("Start Time: %s".format(self.start_dt))

    def stop(self):
        end_dt = dt.datetime.now()
        dur = end_dt - self.start_dt
        print("End Time: %s".format(end_dt))
        print("Total time taken: %s".format(dur))
