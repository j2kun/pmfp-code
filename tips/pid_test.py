from dataclasses import dataclass, replace

import hypothesis
from hypothesis import given
from hypothesis.strategies import floats

from tips.pid import PrincipalIntegralDerviativeController as PID


# A symmetric linear system: the control is directly proportional to the system
# change.
@dataclass(frozen=True)
class SimpleLinearSystem:
    value: float = 0
    time: int = 0

    def run(self, control: int):
        change = control / 10
        new_value = self.value + change
        return replace(self, value=new_value, time=self.time + 1)


# An asymmetric non-linear system: you can turn on the heater to heat fast, or
# leave it off to cool slowly.
@dataclass(frozen=True)
class SimpleHeatingSystem:
    measured_temp: float = 60.0  # fahrenheit
    time: int = 0

    def run(self, control: int):
        new_temp = self.measured_temp + (0.5 if control > 0 else -0.1)
        return replace(self, measured_temp=new_temp, time=self.time + 1)


# A test with fixed, known good parameters
def test_achieves_setpoint():
    setpoint = 72
    pid = PID(kp=5, ki=3, kd=3, setpoint=setpoint)
    system = SimpleHeatingSystem(measured_temp=60.0)

    for _ in range(50):
        control = pid.run(system.measured_temp, dt=1.0)
        system = system.run(control)
        print(f"{control:G}: {system.measured_temp:G}")

    assert abs(system.measured_temp - setpoint) < 2


@given(
    floats(min_value=1, max_value=5, allow_nan=False, allow_infinity=False),
    floats(min_value=0.1, max_value=3, allow_nan=False, allow_infinity=False),
    floats(min_value=0.1, max_value=3, allow_nan=False, allow_infinity=False),
)
@hypothesis.settings(print_blob=True)
def test_achieves_setpoint_parameter_range(kp, ki, kd):
    setpoint = 72
    pid = PID(kp=kp, ki=ki, kd=kd, setpoint=setpoint)
    system = SimpleHeatingSystem(measured_temp=60.0)

    for _ in range(200):
        control = pid.run(system.measured_temp, dt=1.0)
        system = system.run(control)
        print(f"{control:G}: {system.measured_temp:G}")

    assert abs(system.measured_temp - setpoint) < 1


def test_no_derivative_kick():
    pid = PID(kp=0.1, ki=0.1, kd=1, setpoint=100, output_min=-100, output_max=100)
    control = pid.run(0, dt=1)
    assert control < 100


def test_no_integral_windup_from_parameter_changes():
    pid = PID(kp=6, ki=0.6, kd=0.2, setpoint=10, output_min=-100, output_max=100)
    system = SimpleLinearSystem(value=0.0)
    for i in range(20):
        control = pid.run(system.value, dt=1.0)
        system = system.run(control)
        print(f"{control:G}, {system.value:G}")

    pid.ki = 0.3
    next_control = pid.run(system.value, dt=1.0)
    print(f"{next_control:G} - {control:G}")
    assert abs(next_control - control) < 0.2


def test_no_lag_after_setpoint_change_from_integral_windup():
    pid = PID(kp=1, ki=1, kd=1, setpoint=50, output_min=-50, output_max=50)
    system = SimpleLinearSystem(value=0.0)
    # let the integral windup (if it were broken) to > 100
    for i in range(5):
        control = pid.run(system.value, dt=1.0)
        system = system.run(control)
        print(f"{control:G}, {system.value:G}, {pid.integral}")

    assert control == 50
    print("set point change")

    # change the setpoint and assert there is no lag in changing the control
    for i in range(2):
        pid.setpoint = 0
        control = pid.run(system.value, dt=1.0)
        system = system.run(control)
        print(f"{control:G}, {system.value:G}, {pid.integral}")

    assert control < 0
