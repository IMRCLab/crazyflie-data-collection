# Hardware Set Up

* Crazyflie 2.1
    * URI: radio://0/80/2M/E7E7E7E702
* RPM deck
* uSD-card deck (with config.txt as configuration)
* total weight with battery: 37 grams

* Firmware: https://github.com/IMRCLab/crazyflie-firmware/tree/dev-coltrans
* Controller: Lee (number 5)
* firmware_params:
    - pwm:
        -  d00: 0.16609668469985447
        -  d01: 0.0
        -  d10: 0.02297914810965436
        -  d20: 0.0001878280261464696
        -  d11: 0.0

## Experiments

* close blinds
* In Motive, use exposure 2500 us, LEDs off

```
ros2 launch crazyflie launch.py
ros2 run crazyflie_examples figure8
```

## Data 

### eckart{00, 01, 02, 03, 04}

TIMESCALE = 1.0

### eckart{06, 07, 08, 09, 11}

TIMESCALE = 0.9

### eckart{12, 13, 14, 15, 16}

TIMESCALE = 0.8

### eckart{17, 18, 19, 20 ,21}

TIMESCALE = 0.7

### eckart{22. 23, 24, 25, 26}

TIMESCALE = 0.6

### eckart{27, 30, 34, 35, 36}

TIMESCALE = 0.5
