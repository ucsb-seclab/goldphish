# Simple AIMD control system

class BlockThrottle:
    setpoint: float
    additive_increase: float
    mult_decrease: float

    next_val: float
    

    def __init__(self, setpoint: float, initial: float, additive_increase: float = 100) -> None:
        self.setpoint = float(setpoint)
        self.next_val = float(initial)

        # tuning constants
        self.additive_increase = additive_increase
        self.mult_decrease = 0.75

    def observe(self, val: float):
        """
        Observe a measured value
        """
        # if this is over setpoint, decrease
        if val > self.setpoint:
            self.next_val = self.next_val * self.mult_decrease
        else:
            self.next_val = self.next_val + self.additive_increase
    
    def val(self) -> float:
        """
        Get the next value
        """
        return self.next_val

    def val_int_clamp(self, min_, max_) -> int:
        return max(min_, min(max_, round(self.val())))
