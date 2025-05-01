# --- Imports ---
import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# --- Load and Prepare Data ---
csv_path = "nutrition.csv"  # Replace with your actual path if needed
meal_df = pd.read_csv(csv_path)
meal_list = list(meal_df["Dish Name"].dropna().unique()[:8])  # First 8 meals

# --- HMM Setup ---
hidden_states = ["Healthy Mood", "Unhealthy Mood"]

start_prob = {
    "Healthy Mood": 0.5,
    "Unhealthy Mood": 0.5
}

transition_prob = {
    "Healthy Mood": {
        "Healthy Mood": 0.7,
        "Unhealthy Mood": 0.3
    },
    "Unhealthy Mood": {
        "Healthy Mood": 0.2,
        "Unhealthy Mood": 0.8
    }
}

emission_prob = {
    "Healthy Mood": {
        meal_list[0]: 0.2,
        meal_list[1]: 0.2,
        meal_list[2]: 0.2,
        meal_list[5]: 0.2,
        meal_list[3]: 0.05,
        meal_list[4]: 0.05,
        meal_list[6]: 0.05,
        meal_list[7]: 0.05
    },
    "Unhealthy Mood": {
        meal_list[0]: 0.05,
        meal_list[1]: 0.05,
        meal_list[2]: 0.05,
        meal_list[5]: 0.05,
        meal_list[3]: 0.2,
        meal_list[4]: 0.2,
        meal_list[6]: 0.2,
        meal_list[7]: 0.2
    }
}

# --- HMM Meal Sequence Generator ---
def generate_meal_sequence(start_meal, num_steps=7):
    meals = [start_meal]
    moods = []
    current_mood = np.random.choice(hidden_states, p=[start_prob["Healthy Mood"], start_prob["Unhealthy Mood"]])
    moods.append(current_mood)

    for _ in range(num_steps - 1):
        next_mood = np.random.choice(
            hidden_states,
            p=[
                transition_prob[current_mood]["Healthy Mood"],
                transition_prob[current_mood]["Unhealthy Mood"]
            ]
        )
        meal_options = list(emission_prob[next_mood].keys())
        meal_probs = list(emission_prob[next_mood].values())
        next_meal = np.random.choice(meal_options, p=meal_probs)

        moods.append(next_mood)
        meals.append(next_meal)
        current_mood = next_mood

    return meals, moods

# --- Graph Visualization Function ---
def draw_meal_transition_graph_fixed(meals, moods):
    G = nx.DiGraph()
    mood_colors = {"Healthy Mood": "green", "Unhealthy Mood": "red"}
    labeled_nodes = []

    for i, (meal, mood) in enumerate(zip(meals, moods)):
        label = f"Day {i+1}\n{meal}"
        labeled_nodes.append(label)
        G.add_node(label, color=mood_colors[mood], pos=(i, 0))

    for i in range(len(labeled_nodes) - 1):
        G.add_edge(labeled_nodes[i], labeled_nodes[i + 1])

    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[node]['color'] for node in labeled_nodes]

    plt.figure(figsize=(16, 4))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=3500,
        font_size=7,
        font_weight='bold',
        edge_color='gray',
        arrows=True
    )
    plt.title("Meal Plan Transition Graph (Each Node = Meal + Day)")
    plt.show()

# --- Main Interactive Loop ---
def run_meal_planner():
    while True:
        print("\nAvailable meals:")
        for idx, meal in enumerate(meal_list, 1):
            print(f"{idx}. {meal}")

        try:
            choice = int(input("\nPick a starting meal (1-8): "))
            if 1 <= choice <= len(meal_list):
                start_meal = meal_list[choice - 1]
            else:
                print("Please enter a number between 1 and 8.")
                continue
        except ValueError:
            print("Please enter a valid number.")
            continue

        print(f"\nYou chose to start with: {start_meal}")
        meals, moods = generate_meal_sequence(start_meal)

        print("\n🍽️ Your 7-Day Meal Plan:")
        for i in range(len(meals)):
            print(f"Day {i + 1}: {meals[i]}  [{moods[i]}]")

        draw_meal_transition_graph_fixed(meals, moods)

        again = input("\nRun again? (yes/no): ").strip().lower()
        if again != "yes":
            print()
            break

# --- Run the program ---
run_meal_planner()
