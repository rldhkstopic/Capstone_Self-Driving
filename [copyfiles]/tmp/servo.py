import serial

mcu = serial.Serial('COM4', 9600)

while True:
    mcu.write([])
    if mcu.readable():
        res = mcu.readline()
        print(res.decode()[:len(res)-1])
