from tips.pid import PID, SimpleHeatingSystem


def test_achieves_setpoint():
    setpoint = 72
    pid = PID(kp=1.0, ki=1.0, kd=1.0, setpoint=setpoint)
    system = SimpleHeatingSystem(measured_temp=60.0)

    for _ in range(100):
        control = pid.run(system.measured_temp, dt=1.0)
        system = system.run(control)
        print(f"{control:G}: {system.measured_temp:G}")

    assert abs(system.measured_temp - setpoint) < 1
