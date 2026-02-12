from dataclasses import dataclass

@dataclass
class Vehicle:
    name: str
    x_pos: float
    lane: int
    x_vel: float
    y_vel: float

    """
    Purpose: Return whether or not the calling vehicle is in front of the 'other' vehicle
                Used in closest_same_lane function
    """
    def is_ahead_of(self, other):
        return self.x_pos > other.x_pos and self.lane == other.lane

    """
    Purpose: Return how many seconds it will take for the calling vehicle to crash into the given vehicle
                Return infinity if no collision will ever occur given current positions and speeds
    """
    def get_ttc(self, lead_car):
        if not lead_car: return float('inf')

        rel_dst = lead_car.x_pos - self.x_pos
        rel_vel = self.x_vel - lead_car.x_vel

        if rel_dst <= 0: return 0.0
        if rel_vel <= 0: return float('inf')

        return rel_dst / rel_vel

"""
Purpose: From the cars list find the closest vehicle in front of the ego vehicle, if one exists
"""
def closest_same_lane(cars):
    if len(cars) <= 1:
        return None

    ego = cars[0]
    closest_car = None
    min_dist = float('inf')

    # Check all cars in cars list to see which is closest to ego vehicle that is in the same lane
    for car in cars[1:]:
        if car.is_ahead_of(ego):
            dist = car.x_pos - ego.x_pos
            if 0 < dist < min_dist:
                min_dist = dist
                closest_car = car

    return closest_car

def run_test(test_name, correct_statement, check_statement):
    print(f"Test {test_name}:", end=" ")
    try:
        assert correct_statement == check_statement
        print("PASSED")
    except AssertionError:
        print(f"FAILED")
        print(f"   Expected: '{correct_statement}'")
        print(f"   Got:      '{check_statement}'")
    except Exception as e:
        print(f"CRASHED: {e}")

empty_lane = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=20, y_vel=0),
              Vehicle(name="Vehicle 1", x_pos=25, lane=2, x_vel=20, y_vel=0),
              Vehicle(name="Vehicle 2", x_pos=25, lane=3, x_vel=20, y_vel=0)]
run_test("NO CARS IN SAME LANE 1", None, closest_same_lane(empty_lane))
print()

error1 = []
error2 = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=20, y_vel=0)]
run_test("Error 1", None, closest_same_lane(error1))
run_test("Error 2", None, closest_same_lane(error2))
print()

cars1 = [Vehicle(name="Ego vehicle", x_pos=25, lane=1, x_vel=25, y_vel=0),
         Vehicle(name="Vehicle 1", x_pos=0, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 2", x_pos=100, lane=1, x_vel=20, y_vel=0)]
cars2 = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=24, y_vel=0),
        Vehicle(name="Vehicle 1", x_pos=25, lane=1, x_vel=20, y_vel=0),
        Vehicle(name="Vehicle 2", x_pos=100, lane=1, x_vel=20, y_vel=0)]
cars3 = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=21, y_vel=0),
         Vehicle(name="Vehicle 1", x_pos=25, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 2", x_pos=10, lane=1, x_vel=20, y_vel=0)]
cars4 = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=40, y_vel=0),
         Vehicle(name="Vehicle 1", x_pos=25, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 2", x_pos=25, lane=1, x_vel=20, y_vel=0)]
cars5 = [Vehicle(name="Ego vehicle", x_pos=25, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 1", x_pos=-25, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 2", x_pos=-25, lane=1, x_vel=20, y_vel=0)]
cars6 = [Vehicle(name="Ego vehicle", x_pos=50, lane=1, x_vel=20, y_vel=0),
        Vehicle(name="Vehicle 1", x_pos=25, lane=1, x_vel=20, y_vel=0),
        Vehicle(name="Vehicle 2", x_pos=0, lane=1, x_vel=20, y_vel=0)]
cars7 = [Vehicle(name="Ego vehicle", x_pos=50, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 1", x_pos=-25, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 2", x_pos=75, lane=1, x_vel=22, y_vel=0)]

run_test("CLOSEST 1", Vehicle(name="Vehicle 2", x_pos=100, lane=1, x_vel=20, y_vel=0), closest_same_lane(cars1))
run_test("CLOSEST 2", Vehicle(name="Vehicle 1", x_pos=25, lane=1, x_vel=20, y_vel=0), closest_same_lane(cars2))
run_test("CLOSEST 3", Vehicle(name="Vehicle 2", x_pos=10, lane=1, x_vel=20, y_vel=0), closest_same_lane(cars3))
run_test("CLOSEST 4", Vehicle(name="Vehicle 1", x_pos=25, lane=1, x_vel=20, y_vel=0), closest_same_lane(cars4))
run_test("CLOSEST 5", None, closest_same_lane(cars5))
run_test("CLOSEST 6", None, closest_same_lane(cars6))
run_test("CLOSEST 7", Vehicle(name="Vehicle 2", x_pos=75, lane=1, x_vel=22, y_vel=0), closest_same_lane(cars7))
print()

run_test("TIME 1", 15, cars1[0].get_ttc(closest_same_lane(cars1)))
run_test("TIME 2", 6.25, cars2[0].get_ttc(closest_same_lane(cars2)))
run_test("TIME 3", 10, cars3[0].get_ttc(closest_same_lane(cars3)))
run_test("TIME 4", 1.25, cars4[0].get_ttc(closest_same_lane(cars4)))
run_test("TIME 5", float('inf'), cars5[0].get_ttc(closest_same_lane(cars5)))
run_test("TIME 6", float('inf'), cars6[0].get_ttc(closest_same_lane(cars6)))
run_test("TIME 7", float('inf'), cars7[0].get_ttc(closest_same_lane(cars7)))