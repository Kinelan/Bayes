# Asia.py

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def create_asia_model():
    # Define the structure of the Bayesian Network
    model = BayesianNetwork([
        ('VisitAsia', 'Tuberculosis'),
        ('Smoking', 'Lung Cancer'),
        ('Smoking', 'Bronchitis'),
        ('Tuberculosis', 'Tb_or_Ca'),
        ('Lung Cancer', 'Tb_or_Ca'),
        ('Tb_or_Ca', 'XRay'),
        ('Tb_or_Ca', 'Dyspnea'),
        ('Bronchitis', 'Dyspnea')
    ])

    # Define the CPDs (Conditional Probability Distributions)
    cpd_visit_asia = TabularCPD(
        variable='VisitAsia',
        variable_card=2,
        values=[[0.01], [0.99]],
        state_names={'VisitAsia': ['Visit', 'NoVisit']}
    )

    cpd_smoking = TabularCPD(
        variable='Smoking',
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'Smoking': ['Smoking', 'NoSmoking']}
    )

    cpd_tuberculosis = TabularCPD(
        variable='Tuberculosis',
        variable_card=2,
        values=[[0.05, 0.01],
                [0.95, 0.99]],
        evidence=['VisitAsia'],
        evidence_card=[2],
        state_names={'Tuberculosis': ['Present', 'Absent'], 'VisitAsia': ['Visit', 'NoVisit']}
    )

    cpd_lung_cancer = TabularCPD(
        variable='Lung Cancer',
        variable_card=2,
        values=[[0.1, 0.01],
                [0.9, 0.99]],
        evidence=['Smoking'],
        evidence_card=[2],
        state_names={'Lung Cancer': ['Present', 'Absent'], 'Smoking': ['Smoking', 'NoSmoking']}
    )

    cpd_tb_or_ca = TabularCPD(
        variable='Tb_or_Ca',
        variable_card=2,
        values=[[1, 1, 1, 0],
                [0, 0, 0, 1]],
        evidence=['Tuberculosis', 'Lung Cancer'],
        evidence_card=[2, 2],
        state_names={'Tb_or_Ca': ['True', 'False'], 'Tuberculosis': ['Present', 'Absent'], 'Lung Cancer': ['Present', 'Absent']}
    )

    cpd_xray = TabularCPD(
        variable='XRay',
        variable_card=2,
        values=[[0.98, 0.05],
                [0.02, 0.95]],
        evidence=['Tb_or_Ca'],
        evidence_card=[2],
        state_names={'XRay': ['Abnormal', 'Normal'], 'Tb_or_Ca': ['True', 'False']}
    )

    cpd_bronchitis = TabularCPD(
        variable='Bronchitis',
        variable_card=2,
        values=[[0.6, 0.3],
                [0.4, 0.7]],
        evidence=['Smoking'],
        evidence_card=[2],
        state_names={'Bronchitis': ['Present', 'Absent'], 'Smoking': ['Smoking', 'NoSmoking']}
    )

    cpd_dyspnea = TabularCPD(
        variable='Dyspnea',
        variable_card=2,
        values=[[0.9, 0.7, 0.8, 0.1],
                [0.1, 0.3, 0.2, 0.9]],
        evidence=['Tb_or_Ca', 'Bronchitis'],
        evidence_card=[2, 2],
        state_names={'Dyspnea': ['True', 'False'], 'Tb_or_Ca': ['True', 'False'], 'Bronchitis': ['Present', 'Absent']}
    )

    # Add CPDs to the model
    model.add_cpds(cpd_visit_asia, cpd_smoking, cpd_tuberculosis, cpd_lung_cancer, cpd_tb_or_ca, cpd_xray, cpd_bronchitis, cpd_dyspnea)

    # Validate the model
    assert model.check_model()

    return model