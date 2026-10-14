"profiler.py"

from dynamics import meteor_rhs, ground_impact_event, DEFAULT_PARAMS
from cavezzo import initial_state
from scipy.integrate import solve_ivp

def run():
    y0 = initial_state()
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    return solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=DEFAULT_PARAMS),
        t_span=(0.0, 600.0), y0=y0, method="DOP853",
        rtol=1e-12, atol=1e-14, max_step=5.0,
        events=ground_impact_event,
    )

if __name__ == "__main__":
    for _ in range(20):
        run()