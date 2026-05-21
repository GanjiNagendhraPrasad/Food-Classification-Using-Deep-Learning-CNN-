import redis
import json
# Connect Redis
r = redis.Redis(
    host='127.0.0.1',
    port=6379,
    decode_responses=True
)
# # Food name to search
# food_name = "pizza"
# # Get complete food data
# data = r.get("food_details")
# # Check data exists
# if data is None:
#     print("No data found in Redis")
# else:
#     food_data = json.loads(data)
#     # Check food exists
#     if food_name in food_data:
#         print("Food Name:", food_name)
#         print("Nutrition Data:")
#         print(food_data[food_name])
#     else:
#         print("Food not found")

for key in r.keys("*"):
    value = json.loads(r.get(key))
    print(key, ":", value)