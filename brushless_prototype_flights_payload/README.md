# Hardware Set Up

* Crazyflie 2.1
    * URI: radio://0/80/2M/E7E7E7E7E7
* RPM deck
* uSD-card deck (with config.txt as configuration)
* total weight with battery: 44.4 grams
    * payload: 4.7 grams
    * UAV (with battery): 39.7

* Firmware: https://github.com/IMRCLab/crazyflie-firmware/tree/dev-coltrans
* Controller: Lee (number 5)
* firmware_params:
      pwm:
        d00: 0.166609668469985447
        d01: 0.0
        d10: 0.02297914810965436
        d20: 0.0001878280261464696
        d11: 0.0
      ctrlLee:
        mass: 0.0444

## Experiments

* close blinds
* In Motive, use exposure 2000 us, LEDs off, 100 Hz
* height: 0.75 (otherwise loses track!)

```
QT_QPA_PLATFORM=xcb ros2 launch crazyflie launch.py gui:=False rviz:=True
ros2 run crazyflie_examples figure8
```

## Data 

### eckart{06, 07, 08, 09, 10}

TIMESCALE = 1.0

### eckart{11, 12, 13, 14, 15}

TIMESCALE = 0.9

### eckart{17, 18, 19, 20, 21}

TIMESCALE = 0.8

### eckart{22, 24, 26, 27, 28}

TIMESCALE = 0.7

### eckart{29, 30, 32, 33, 34}

TIMESCALE = 0.6

### eckart{43, 45, 46, 47, 48}

sideways
HEIGHT = 0.75
TIME = 1.5
pt1 = [0, 0.5, HEIGHT]
pt2 = [0, -0.5, HEIGHT]

### eckart{51, 52, 53, 54, 56}

sideways
HEIGHT = 0.75
TIME = 1.3
pt1 = [0, 0.5, HEIGHT]
pt2 = [0, -0.5, HEIGHT]

### eckart{57, 58, 59, 60, 61}

sideways
HEIGHT = 0.75
TIME = 1.2
pt1 = [0, 0.5, HEIGHT]
pt2 = [0, -0.5, HEIGHT]

### eckart{62, 63, 64, 65, 66}

sideways
HEIGHT = 0.75
TIME = 1.1
pt1 = [0, 0.5, HEIGHT]
pt2 = [0, -0.5, HEIGHT]

### eckart{67, 68, 69, 70, 71}

sideways
HEIGHT = 0.75
TIME = 1.5
pt1 = [0.5, 0.0, HEIGHT]
pt2 = [-0.5, 0.0, HEIGHT]

### eckart{72, 73, 74, 75, 76}

sideways
HEIGHT = 0.75
TIME = 1.3
pt1 = [0.5, 0.0, HEIGHT]
pt2 = [-0.5, 0.0, HEIGHT]

### eckart{77, 78, 79, 80, 81}

sideways
HEIGHT = 0.75
TIME = 1.2
pt1 = [0.5, 0.0, HEIGHT]
pt2 = [-0.5, 0.0, HEIGHT]

### eckart{82, 83, 84, 85, 86}

sideways
HEIGHT = 0.75
TIME = 1.1
pt1 = [0.5, 0.0, HEIGHT]
pt2 = [-0.5, 0.0, HEIGHT]