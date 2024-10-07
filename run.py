#run.py

from pgmpy.inference import VariableElimination
import itertools
#from Animals import create_animals_model
from Asia import create_asia_model


# Create the model
#model = create_animals_model()
model = create_asia_model()

# Automatically get the list of all variables (nodes) in the model
variables = model.nodes()

# Perform inference
inference = VariableElimination(model)

# Querying the joint probability for the entire network using automatically fetched variables
joint_prob = inference.query(variables=variables)


# Helper function to map index to state names with variable labels
def get_state_description(factor, index):
    # Get all possible combinations of states
    states = list(itertools.product(*[factor.state_names[var] for var in factor.variables]))
    selected_state = states[index]

    # Pair variable names with their states and return them as formatted strings
    return ', '.join([f"{var}: {state}" for var, state in zip(factor.variables, selected_state)])


# Write only significant joint probabilities to file with detailed state descriptions
#with open('animals_results_filtered_detailed.txt', 'w') as file:
with open('asia_results_filtered_detailed.txt', 'w') as file:
    file.write("Significant Joint Probabilities with State Descriptions:\n")
    for idx, prob in enumerate(joint_prob.values.flatten()):
        if prob > 0:  # Only write non-zero probabilities
            state_description = get_state_description(joint_prob, idx)
            file.write(f"  State: {state_description}, Probability: {prob:.10f}\n")