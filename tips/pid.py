from dataclasses import dataclass, replace


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    setpoint: float
    last_measurement: float = 0.0
    integral: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    # One could also set a constant dt and ensure the PID is called in regular
    # intervals. This would be useful for avoiding extra multiplications and
    # divisions when running on a microcontroller.

    def run(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        # Put the tuning constant in the accumulation to avoid integral windup
        # impacting online parameter changes.
        # Naively, this would be self.integral += error * dt, with self.ki
        # in the output calculation.
        self.integral += self.ki * error * dt
        # Clamp to avoid lag from integral windup
        self.integral = clamp(self.integral, self.output_min, self.output_max)

        # Assume setpoint remains constant and ignore instantaneous setpoint
        # changes to avoid derivative kick.
        # Naively, this would be derivative = (error - self.last_error) / dt
        derivative = (measurement - self.last_measurement) / dt

        output = self.kp * error + self.integral + self.kd * derivative
        self.last_measurement = measurement
        return clamp(output, self.output_min, self.output_max)


@dataclass(frozen=True)
class SimpleHeatingSystem:
    measured_temp: float = 60.0  # fahrenheit
    time: int = 0

    def run(self, control: int):
        new_temp = self.measured_temp + 1 if control > 0 else self.measured_temp - 0.1
        return replace(self, measured_temp=new_temp, time=self.time + 1)
