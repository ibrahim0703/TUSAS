# simulate a move
action = "right"
print(f"\n-- Action taken: {action.upper()}---")
next_state = env.step(action)
print(f"New State: {next_state}")
# check distance condition ın range r radius circle.
dist = next_state["distance_to_target"]
if dist<1 :
    print("\n[INFO] Target Destroyed !")
else:
    print(f"\n[INFO] Target is still {dist} units away.")
