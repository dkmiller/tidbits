import badger2040
from badger2040 import WIDTH


class EventHandler():
    def __init__(self, display, speed = badger2040.UPDATE_NORMAL):
        self.display = display
        display.set_update_speed(speed)

    def setup(self):
        self.display.clear()
        self.display.set_font("sans")
        self.display.set_thickness(3)
        self.display.set_pen(0)
        self.display.rectangle(0, 0, WIDTH, 16)
        self.display.set_pen(15)

    def wait(self):
        while True:
            if self.display.pressed(badger2040.BUTTON_A):
                self.on_button("a")
                self.on_button_a()
            elif self.display.pressed(badger2040.BUTTON_B):
                self.on_button("b")
                self.on_button_b()
            elif self.display.pressed(badger2040.BUTTON_C):
                self.on_button("c")
                self.on_button_c()
            elif self.display.pressed(badger2040.BUTTON_DOWN):
                self.on_button("down")
                self.on_button_down()
            elif self.display.pressed(badger2040.BUTTON_UP):
                self.on_button("up")
                self.on_button_up()

    def on_button(self, button):
        pass

    def on_button_a(self):
        pass

    def on_button_b(self):
        pass

    def on_button_c(self):
        pass

    def on_button_down(self):
        pass

    def on_button_up(self):
        pass

class MyEventHandler(EventHandler):
    def __init__(self, display):
        super().__init__(display)
        self.brightness = 0.0

    def set_led(self):
        # 0 (off) .. 1 (max)
        self.display.led(int(self.brightness * 255))
        self.display.text("Brightness " + str(self.brightness), 20, 20, WIDTH, 1)
        self.display.update()

    def on_button(self, button):
        self.display.text("Button " + button, 20, 60, WIDTH, 1)
        self.display.update()

    def on_button_down(self):
        self.brightness = max(self.brightness - 0.1, 0)
        self.set_led()

    def on_button_up(self):
        self.brightness = min(self.brightness + 0.1, 1)
        self.set_led()



# Create a new Badger and set it to update at normal speed.
# _display = badger2040.Badger2040()

handler = MyEventHandler(badger2040.Badger2040())
handler.setup()
handler.wait()
