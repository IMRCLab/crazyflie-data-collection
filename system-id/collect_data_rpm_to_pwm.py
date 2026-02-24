"""
Ramp motors and collect data using the system-id deck
"""
import argparse
import logging
import time
import yaml
from threading import Thread
import random
import copy
import math

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.localization import Localization


class CollectData:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """
 
        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.fully_connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True
        self.battery_low = 0 # counter, how often the battery was below a threshold
        self.last_data = None

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        self._file = open("data.csv", "w+")
        self._file.write("pwm,vbat[V],rpm1,rpm2,rpm3,rpm4\n");

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='data', period_in_ms=1000)
        # self._lg_stab.add_variable('loadcell.weight', 'float')
        self._lg_stab.add_variable('motor.m1', 'uint16_t')
        self._lg_stab.add_variable('pm.vbatMV', 'uint16_t')
        self._lg_stab.add_variable('rpm.m1', 'uint16_t')
        self._lg_stab.add_variable('rpm.m2', 'uint16_t')
        self._lg_stab.add_variable('rpm.m3', 'uint16_t')
        self._lg_stab.add_variable('rpm.m4', 'uint16_t')
        # self._lg_stab.add_variable('asc37800.v_mV', 'int16_t')
        # self._lg_stab.add_variable('asc37800.i_mA', 'int16_t')
        # self._lg_stab.add_variable('asc37800.p_mW', 'int16_t')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        Thread(target=self._ramp_motors).start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback froma the log API when data arrives"""
        print('[%d][%s]: %s' % (timestamp, logconf.name, data))
        # once we receive something new, write out the last data (to allow the motors to settle)
        if self.last_data is not None and data['motor.m1'] != self.last_data['motor.m1']:
            self._file.write("{},{},{},{},{},{}\n".format(
                self.last_data['motor.m1'],
                self.last_data['pm.vbatMV']/ 1000,
                self.last_data['rpm.m1'],
                self.last_data['rpm.m2'],
                self.last_data['rpm.m3'],
                self.last_data['rpm.m4']))
            self._file.flush()

        # check if the voltage is low
        if data['motor.m1'] < 40000 and data['pm.vbatMV']/ 1000 < 3.5:
            self.battery_low = self.battery_low + 1

        # store a copy
        self.last_data = copy.deepcopy(data)

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the speficied address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

    def _ramp_motors(self):
        time_step = 3 #0.1

        localization = Localization(self._cf)

        # self._cf.param.set_value('motor.batCompensation', 0)
        self._cf.param.set_value('motorPowerSet.m1', 0)
        self._cf.param.set_value('motorPowerSet.m2', 0)
        self._cf.param.set_value('motorPowerSet.m3', 0)
        self._cf.param.set_value('motorPowerSet.m4', 0)

        self._cf.param.set_value('motorPowerSet.enable', 1)
        # self._cf.param.set_value('system.forceArm', 1)
        # iters = 0
        thrust = 0
        while self.is_connected and self.battery_low < 3: #thrust >= 0:
            old_thrust = thrust
            while True:
                thrust = int(random.uniform(20000, 65536))
                if math.fabs(thrust-old_thrust) > 1500:
                    break
            print(thrust)
            localization.send_emergency_stop_watchdog()
            self._cf.param.set_value('motorPowerSet.m1', str(thrust))
            self._cf.param.set_value('motorPowerSet.m2', str(thrust))
            self._cf.param.set_value('motorPowerSet.m3', str(thrust))
            self._cf.param.set_value('motorPowerSet.m4', str(thrust))

            time.sleep(time_step)
            # iters += 1

        self._cf.param.set_value('motorPowerSet.enable', 0)
        time.sleep(3.0)


        # self._cf.commander.send_setpoint(0, 0, 0, 0)
        # Make sure that the last packet leaves before the link is closed
        # since the message queue is not flushed before closing
        # time.sleep(0.1)
        self._cf.close_link()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7", help="URI of Crazyflie")
    args = parser.parse_args()

    # Only output errors from the logging framework
    logging.basicConfig(level=logging.ERROR)

    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    # collect data
    le = CollectData(args.uri)

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    while le.is_connected:
        time.sleep(1)
