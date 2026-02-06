"""
Purpose: From the cars list find the closest vehicle in front of the ego vehicle, if one exists
"""
def closest_same_lane(cars):
    if not cars:
        return "No cars currently exist"

    if len(cars) == 1:
        return "Only the ego vehicle exists"

    # Mark down ego lane and position for future access
    ego_lane = cars[0][2]
    ego_pos = cars[0][1]

    # Filter vehicles out of lane_cars list that aren't in the same lane as the ego vehicle
    lane_cars = [car for car in cars if car[2] == ego_lane]

    if(len(lane_cars) == 1):
        return f"There is no car close to the ego vehicle in lane {ego_lane}"

    vehicle_index = -1
    min_dist = float('inf')
        
    for i in range(1, len(lane_cars)):
        dist = lane_cars[i][1] - ego_pos
        if dist > 0 and dist < min_dist:
            min_dist = dist
            vehicle_index = i

    if min_dist == float('inf'):
        return f"There is no car ahead of the ego vehicle that is close to it in lane {ego_lane}"

    return f"The closest vehicle to the ego vehicle in lane {ego_lane} is {lane_cars[vehicle_index][0]} at position x = {lane_cars[vehicle_index][1]}"

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

error1 = []
error2 = [["Vehicle 1", 25, 1, 20, 0]]
run_test("Error 1", "No cars currently exist", closest_same_lane(error1))
run_test("Error 2", "Only the ego vehicle exists", closest_same_lane(error2))

empty_lane = [["Ego vehicle", 0, 1, 20, 0], ["Vehicle 1", 25, 2, 20, 0], ["Vehicle 2", 25, 3, 20, 0]]
run_test("NO CARS IN SAME LANE 1", "There is no car close to the ego vehicle in lane 1", closest_same_lane(empty_lane))


cars1 = [["Ego vehicle", 25, 1, 20, 0], ["Vehicle 1", 0, 1, 20, 0], ["Vehicle 2", 100, 1, 20, 0]]
cars2 = [["Ego vehicle", 0, 1, 20, 0], ["Vehicle 1", 25, 1, 20, 0], ["Vehicle 2", 100, 1, 20, 0]]
cars3 = [["Ego vehicle", 0, 1, 20, 0], ["Vehicle 1", 25, 1, 20, 0], ["Vehicle 2", 10, 1, 20, 0]]
cars4 = [["Ego vehicle", 0, 1, 20, 0], ["Vehicle 1", 25, 1, 20, 0], ["Vehicle 2", 25, 1, 20, 0]]
cars5 = [["Ego vehicle", 0, 1, 20, 0], ["Vehicle 1", -25, 2, 20, 0], ["Vehicle 2", -25, 3, 20, 0]]
cars6 = [["Ego vehicle", 50, 1, 20, 0], ["Vehicle 1", 25, 1, 20, 0], ["Vehicle 2", 0, 1, 20, 0]]
cars7 = [["Ego vehicle", 50, 1, 20, 0], ["Vehicle 1", -25, 1, 20, 0], ["Vehicle 2", 75, 1, 20, 0]]

run_test("CLOSEST 1", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 2 at position x = 100", closest_same_lane(cars1))
run_test("CLOSEST 2", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 1 at position x = 25", closest_same_lane(cars2))
run_test("CLOSEST 3", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 2 at position x = 10", closest_same_lane(cars3))
run_test("CLOSEST 4", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 1 at position x = 25", closest_same_lane(cars4))
run_test("CLOSEST 5", "There is no car close to the ego vehicle in lane 1", closest_same_lane(cars5))
run_test("CLOSEST 6", "There is no car ahead of the ego vehicle that is close to it in lane 1", closest_same_lane(cars6))
run_test("CLOSEST 7", "The closest vehicle to the ego vehicle in lane 1 is Vehicle 2 at position x = 75", closest_same_lane(cars7))