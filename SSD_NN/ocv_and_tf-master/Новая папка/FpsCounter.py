from queue import Queue
import datetime
class FpsCounter():
    def __init__(self):
        self.fps_que = Queue()
        self.fps_que_size = 50
        for val in [0] * self.fps_que_size: self.fps_que.put(val)
        self.summ_fps = 0.0
        self.fps = 0.0

    def start(self):
        self.start_time = datetime.datetime.now()

    def checkpoint(self):
        now_time = datetime.datetime.now()
        self.fps = (1.0 / ((now_time - self.start_time).total_seconds() + 0.00005))
        self.summ_fps = self.summ_fps - self.fps_que.get() + self.fps
        self.avFPS = self.summ_fps / self.fps_que_size
        self.fps_que.put(self.fps)

    def __str__(self):
        return ("FPS " + "{:2.2f}" + " avFPS {:2.2f}").format(self.fps, self.avFPS)

