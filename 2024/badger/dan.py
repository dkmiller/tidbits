import time

import badger2040
from badger2040 import WIDTH


class EventHandler():
    def __init__(self, display):
        self.display = display

    def wait(self):
        while True:
            if display.pressed(badger2040.BUTTON_DOWN):
                self.on_button_down()
            elif display.pressed(badger2040.BUTTON_UP):
                self.on_button_up()

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

    def on_button_down(self):
        self.brightness = max(self.brightness - 0.1, 0)
        self.set_led()

    def on_button_up(self):
        self.brightness = min(self.brightness + 0.1, 1)
        self.set_led()


def set_led(display, brightness):
    # 0 (off) .. 1 (max)
    display.led(int(brightness * 255))


# Create a new Badger and set it to update at normal speed.
display = badger2040.Badger2040()

set_led(display, 0.9)

display.set_update_speed(badger2040.UPDATE_NORMAL)

SPACING = 20

import machine
# (year, month, mday, hour, minute, second, weekday, yearday)
# machine.RTC().datetime((2024, 12, 17, 2, 37, 14, 0, 352))

(year, month, mday, hour, minute, second, weekday, yearday) = time.localtime()
_foo = "{0}/{1} {2}:{3}".format(month, mday, hour, minute)

# str(display.pressed_any()) + 

if display.pressed(badger2040.BUTTON_A):
    message = "Pressed A"
    set_led(display, 0.1)
if display.pressed(badger2040.BUTTON_B):
    message = "Pressed B"
    set_led(display, 0.5)
if display.pressed(badger2040.BUTTON_C):
    message = "Pressed B"
    set_led(display, 0.9)
if display.pressed(badger2040.BUTTON_UP):
    message = "Up!!"
    set_led(display, 1)
if display.pressed(badger2040.BUTTON_DOWN):
    message = "down..."
    set_led(display, 0)
else:
    message = "no button"

display.clear()
display.set_font("sans")
display.set_thickness(3)
display.set_pen(0)
display.rectangle(0, 0, WIDTH, 16)
display.set_pen(15)

handler = MyEventHandler(display)
handler.wait()

# Reset display and show message
display.text(_foo, SPACING, SPACING, WIDTH, 1)
display.text(str(display.pressed_any()), SPACING, 3 * SPACING, WIDTH, 1)
display.text("|Poop status?|", SPACING, 5 * SPACING, WIDTH, 1)
display.update()
