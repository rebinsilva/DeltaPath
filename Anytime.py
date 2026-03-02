import signal

PRINT = False


class AnytimeAlgorithm:
    def __init__(self, timeout=None):
        self._result = None
        if timeout is None:
            timeout = 10000
        self.timeout = timeout

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value
        if PRINT:
            if self._result is None:
                print(None, ":", signal.getitimer(signal.ITIMER_VIRTUAL)[0])
            else:
                print(len(self._result),":",  self.timeout - signal.getitimer(signal.ITIMER_VIRTUAL)[0])
    
    def set_marker(self, txt):
        if PRINT:
            print(txt, ":", self.timeout - signal.getitimer(signal.ITIMER_VIRTUAL)[0])

    def __str__(self):
        return str(self.__class__.__name__)

    def _run(self):
        pass

    def signal_handler(self, sig, frame):
        raise TimeoutError()

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        signal.signal(signal.SIGVTALRM, self.signal_handler)
        signal.setitimer(signal.ITIMER_VIRTUAL, self.timeout)

        try:
            result = self._run(*args, **kwargs)
        except TimeoutError:
            print("TIMEOUT")
            result = self.result

        self.result = self.result
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
        self.result = None

        return result

class AnytimeAlgorithm2:
    def __init__(self, timeout=None):
        self._result = None

    def set_marker(self, txt):
        return

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        result = self._run(*args, **kwargs)
        return result