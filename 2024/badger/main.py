import badger2040
import pngdec

display = badger2040.Badger2040()
png = pngdec.PNG(display.display)

display.led(128)
display.clear()

try:
    png.open_file("badge.png")
    png.decode()
except (OSError, RuntimeError):
    print("Badge background error")

display.update()
