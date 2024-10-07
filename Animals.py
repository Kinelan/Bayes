#Animals.py
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def create_animals_model():
    #Define the structure of the Bayesian Network
    model = BayesianNetwork([
        ('Animal', 'Environment'),
        ('Animal', 'HasShell'),
        ('Animal', 'BearsYoungAs'),
        ('Animal', 'Class'),
        ('Class', 'WarmBlooded'),
        ('Class', 'BodyCovering')
    ])

    #Define the CPDs (Conditional Probability Distributions)
    cpd_animal = TabularCPD(
        variable = 'Animal',
        variable_card = 5,
        values = [[0.2], [0.2], [0.2], [0.2], [0.2]],
        state_names = {'Animal': ['Monkey', 'Penguin', 'Platypus', 'Robin', 'Turtle']}
    )

    cpd_environment = TabularCPD(
        variable = 'Environment',
        variable_card = 3,
        values = [[0, 0, 0, 0.5, 0],
                 [1, 0.5, 0, 0.5, 0.5],
                 [0, 0.5, 1, 0, 0.5]],
        evidence = ['Animal'],
        evidence_card = [5],
        state_names = {'Environment': ['Air', 'Land', 'Water'], 'Animal': ['Monkey', 'Penguin', 'Platypus', 'Robin', 'Turtle']}
    )

    cpd_has_shell = TabularCPD(
        variable = 'HasShell',
        variable_card = 2,
        values = [[0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 0]],
        evidence = ['Animal'],
        evidence_card = [5],
        state_names = {'HasShell': ['True', 'False'], 'Animal': ['Monkey', 'Penguin', 'Platypus', 'Robin', 'Turtle']}
    )

    cpd_bears_young_as = TabularCPD(
        variable = 'BearsYoungAs',
        variable_card = 2,
        values = [[1, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1]],
        evidence = ['Animal'],
        evidence_card = [5],
        state_names = {'BearsYoungAs': ['Live', 'Eggs'], 'Animal': ['Monkey', 'Penguin', 'Platypus', 'Robin', 'Turtle']}
    )

    cpd_class = TabularCPD(
        variable = 'Class',
        variable_card = 3,
        values = [[0, 1, 0, 1, 0],
                 [1, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1]],
        evidence = ['Animal'],
        evidence_card = [5],
        state_names = {'Class': ['Bird', 'Mammal', 'Reptile'], 'Animal': ['Monkey', 'Penguin', 'Platypus', 'Robin', 'Turtle']}
    )

    cpd_warm_blooded = TabularCPD(
        variable = 'WarmBlooded',
        variable_card = 2,
        values = [[1, 1, 0],
                 [0, 0, 1]],
        evidence = ['Class'],
        evidence_card = [3],
        state_names = {'WarmBlooded': ['True', 'False'], 'Class': ['Bird', 'Mammal', 'Reptile']}
    )

    cpd_body_covering = TabularCPD(
        variable = 'BodyCovering',
        variable_card = 3,
        values = [[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, 1]],
        evidence = ['Class'],
        evidence_card = [3],
        state_names = {'BodyCovering': ['Fur', 'Feathers', 'Scales'], 'Class': ['Bird', 'Mammal', 'Reptile']}
    )

    #Add CPDs to the model
    model.add_cpds(cpd_animal, cpd_environment, cpd_has_shell, cpd_bears_young_as, cpd_class, cpd_warm_blooded, cpd_body_covering)

    #Validate the model
    assert model.check_model()

    return model