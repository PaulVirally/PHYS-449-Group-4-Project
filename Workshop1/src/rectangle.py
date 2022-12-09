from shapes import ShapesBase
# inherit from ShapesBase

class Rectangle(ShapesBase):

    def __init__(self, a, b):
        # two parameters define a rectangle
        self.a = a
        self.b = b

    def area(self):
        # Area of a rectangle
        return self.a * self.b

    def perimeter(self):
        # perimeter of rectangle
        return 2*(self.a + self.b)