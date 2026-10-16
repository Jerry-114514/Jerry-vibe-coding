"""Independent CPU replay of all minimal-loop checkpoints and scored outputs."""
from verify_route_b_stage_outputs import verify

if __name__ == '__main__':
    print(verify('minimal'))
