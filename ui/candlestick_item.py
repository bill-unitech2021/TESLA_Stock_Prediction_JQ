import pyqtgraph as pg

class CandlestickItem(pg.GraphicsObject):
    def __init__(self, x, open, close, high, low, width=0.6, color='r'):
        pg.GraphicsObject.__init__(self)
        self.x = x
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.width = width
        self.color = color
        self.picture = None
        self.generatePicture()
        
    # ... Rest of CandlestickItem implementation ... 