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


# Plot the PID control in a simulated environment
if __name__ == "__main__":
    from dataclasses import dataclass, replace

    import matplotlib.pyplot as plt

    @dataclass(frozen=True)
    class SimpleLinearSystem:
        value: float = 0
        time: int = 0

        def run(self, control: float):
            change = control / 20
            new_value = self.value + change
            return replace(self, value=new_value, time=self.time + 1)

    samples = 200
    xs = list(range(samples))
    dt = 0.1
    initial = 0
    setpoint = 10
    PID = PrincipalIntegralDerviativeController

    pids = [
        PID(kp=5.0, ki=0.5, kd=0.5, setpoint=setpoint, output_min=-100, output_max=100),
        PID(kp=1.0, ki=1.0, kd=0.5, setpoint=setpoint, output_min=-100, output_max=100),
        PID(kp=0.3, ki=0.0, kd=0.5, setpoint=setpoint, output_min=-100, output_max=100),
    ]
    styles = [
        "dashed",
        "dotted",
        "dashdot",
    ]
    plt.subplots(figsize=(8, 4))
    plt.axhline(
        y=setpoint,
        color="black",
        linestyle="solid",
        label="setpoint",
        linewidth=1,
    )

    for pid, style in zip(pids, styles):
        system = SimpleLinearSystem(value=initial)
        control_values = []
        system_values = []
        for _ in range(samples):
            control = pid.run(system.value, dt=dt)
            system = system.run(control)
            control_values.append(control)
            system_values.append(system.value)

        plt.plot(
            xs,
            system_values,
            linewidth=2,
            label=f"kp={pid.kp:G}, ki={pid.ki:G}, kd={pid.kd:G}",
            linestyle=style,
        )

    plt.xlabel("timestep")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
