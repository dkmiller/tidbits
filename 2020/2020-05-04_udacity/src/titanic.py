import pandas as pd
from src import common


def sex_as_int(sex: str) -> str:
    if sex == 'male':
        return 1
    return 0


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['cabin', 'home.dest', 'name', 'ticket'])
    df['sex'] = df.apply(lambda r: sex_as_int(r['sex']))
    return df


def titanic_csv() -> str:
    return common.path_in_data_directory(
        'titanic.csv',
        'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
    )


def raw_data() -> pd.DataFrame:
    csv = titanic_csv()
    return pd.read_csv(csv)
