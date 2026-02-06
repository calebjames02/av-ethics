from dataclasses import dataclass

@dataclass
class Vehicle:
    name: str
    x_pos: float
    lane: int
    x_vel: float
    y_vel: float

"""
Purpose: From the cars list find the closest vehicle in front of the ego vehicle, if one exists
"""
def closest_same_lane(cars):
    if not cars:
        return "No cars currently exist"

    if len(cars) == 1:
        return "Only the ego vehicle exists"

    # Mark down ego lane and position for future access
    ego_lane = cars[0].lane
    ego_pos = cars[0].x_pos

    # Filter vehicles out of lane_cars list that aren't in the same lane as the ego vehicle
    lane_cars = [car for car in cars if car.lane == ego_lane]

    if(len(lane_cars) == 1):
        return f"There is no car close to the ego vehicle in lane {ego_lane}"

    closest_car = None
    min_dist = float('inf')
        
    for car in lane_cars[1:]:
        dist = car.x_pos - ego_pos
        if dist > 0 and dist < min_dist:
            min_dist = dist
            closest_car = car

    if min_dist == float('inf'):
        return f"There is no car ahead of the ego vehicle that is close to it in lane {ego_lane}"

    return f"The closest vehicle to the ego vehicle in lane {ego_lane} is {closest_car.name} at position x = {closest_car.x_pos}"

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
run_test("NO CARS IN SAME LANE 1", "There is no car close to the ego vehicle in lane 1", closest_same_lane(empty_lane))
print()

error1 = []
error2 = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=20, y_vel=0)]
run_test("Error 1", "No cars currently exist", closest_same_lane(error1))
run_test("Error 2", "Only the ego vehicle exists", closest_same_lane(error2))
print()


cars1 = [Vehicle(name="Ego vehicle", x_pos=25, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 1", x_pos=0, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 2", x_pos=100, lane=1, x_vel=20, y_vel=0)]
cars2 = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=20, y_vel=0),
        Vehicle(name="Vehicle 1", x_pos=25, lane=1, x_vel=20, y_vel=0),
        Vehicle(name="Vehicle 2", x_pos=100, lane=1, x_vel=20, y_vel=0)]
cars3 = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 1", x_pos=25, lane=1, x_vel=20, y_vel=0),
         Vehicle(name="Vehicle 2", x_pos=10, lane=1, x_vel=20, y_vel=0)]
cars4 = [Vehicle(name="Ego vehicle", x_pos=0, lane=1, x_vel=20, y_vel=0),
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
         Vehicle(name="Vehicle 2", x_pos=75, lane=1, x_vel=20, y_vel=0)]

run_test("CLOSEST 1", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 2 at position x = 100", closest_same_lane(cars1))
run_test("CLOSEST 2", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 1 at position x = 25", closest_same_lane(cars2))
run_test("CLOSEST 3", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 2 at position x = 10", closest_same_lane(cars3))
run_test("CLOSEST 4", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 1 at position x = 25", closest_same_lane(cars4))
run_test("CLOSEST 5", "There is no car ahead of the ego vehicle that is close to it in lane 1", closest_same_lane(cars5))
run_test("CLOSEST 6", "There is no car ahead of the ego vehicle that is close to it in lane 1", closest_same_lane(cars6))
run_test("CLOSEST 7", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 2 at position x = 75", closest_same_lane(cars7))