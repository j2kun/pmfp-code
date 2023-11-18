from dataclasses import dataclass


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


@dataclass
class PrincipalIntegralDerviativeController:
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

        # Clamp to avoid lag from integral windup when setpoint changes.
        self.integral = clamp(self.integral, self.output_min, self.output_max)

        # Assume setpoint remains constant and ignore instantaneous setpoint
        # changes to avoid derivative kick.
        # Naively, this would be derivative = (error - self.last_error) / dt
        # But because error is linear, dError = dSetpoint - dMeasurement
        # and we assume dSetpoint = 0 to use this trick.
        derivative = (measurement - self.last_measurement) / dt

        output = self.kp * error + self.integral + self.kd * derivative
        self.last_measurement = measurement
        return clamp(output, self.output_min, self.output_max)
