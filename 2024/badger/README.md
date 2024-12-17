# Badger scripting

## Setup & running

```bash
pip install -r requirements.txt

# Propagate system time.
mpremote connect /dev/cu.usbmodem1101 rtc --set

./run.sh
```

## Navigation

```bash
mpremote connect /dev/cu.usbmodem1101 ls /

mpremote connect /dev/cu.usbmodem1101 cat /main.py
```

## Links

- [badger / .. / hello.py](https://github.com/badger/home/blob/main/examples/hello/hello.py)
- [mpremote](https://docs.micropython.org/en/latest/reference/mpremote.html)
- [Viper IDE](https://viper-ide.org/)
- [Badger 2040: Reference](https://github.com/badger/home/blob/main/2040reference.md)
- https://docs.micropython.org/en/latest/library/time.html
- No networking: https://github.com/pimoroni/badger2040/blob/main/docs/reference.md#differences-between-badger-2040-w-and-badger-2040
