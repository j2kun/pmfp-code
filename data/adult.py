import os

name = "adult"

employers = (
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
)
maritals = (
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
)
occupations = (
    "Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces",
)
races = ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")
sexes = ("Female", "Male")
countries = (
    "United-States",
    "Cambodia",
    "England",
    "Puerto-Rico",
    "Canada",
    "Germany",
    "Outlying-US(Guam-USVI-etc)",
    "India",
    "Japan",
    "Greece",
    "South",
    "China",
    "Cuba",
    "Iran",
    "Honduras",
    "Philippines",
    "Italy",
    "Poland",
    "Jamaica",
    "Vietnam",
    "Mexico",
    "Portugal",
    "Ireland",
    "France",
    "Dominican-Republic",
    "Laos",
    "Ecuador",
    "Taiwan",
    "Haiti",
    "Columbia",
    "Hungary",
    "Guatemala",
    "Nicaragua",
    "Scotland",
    "Thailand",
    "Yugoslavia",
    "El-Salvador",
    "Trinadad&Tobago",
    "Peru",
    "Hong",
    "Holand-Netherlands",
)


feature_names = (
    ("age", "sex")
    + employers
    + ("education",)
    + maritals
    + occupations
    + races
    + ("capital_gain", "capital_loss", "hr_per_week")
    + countries
)


def dataset_paths(datasetName):
    prefix = os.path.join(os.path.dirname(__file__), datasetName)
    return prefix + ".train", prefix + ".test"


def vectorize(value, values):
    return [int(v == value) for v in values]


def process_line(line):
    values = line.strip().split(", ")
    (
        age,
        employer,
        _,
        _,
        education,
        marital,
        occupation,
        _,
        race,
        sex,
        capital_gain,
        capital_loss,
        hr_per_week,
        country,
        income,
    ) = values

    point = (
        [int(age), 0 if sex == "Female" else 1]
        + vectorize(employer, employers)
        + [int(education)]
        + vectorize(marital, maritals)
        + vectorize(occupation, occupations)
        + vectorize(race, races)
        + [int(capital_gain), int(capital_loss), int(hr_per_week)]
        + vectorize(country, countries)
    )
    label = 1 if income[0] == ">" else -1

    return tuple(point), label


def load():
    train_path, test_path = dataset_paths("adult")

    with open(train_path) as infile:
        trainingData = [process_line(line) for line in infile]

    with open(test_path) as infile:
        testData = [process_line(line) for line in infile]

    return trainingData, testData
