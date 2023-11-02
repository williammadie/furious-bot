import os


class DataAccessor:
    @staticmethod
    def data_dirpath() -> str:
        project_dir_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_dir_path, "data")

    @staticmethod
    def intents_filepath() -> str:
        return os.path.join(DataAccessor.data_dirpath(), "intents.json")
    
    @staticmethod
    def model_data_filepath() -> str:
        return os.path.join(DataAccessor.data_dirpath(), "model_data.pth")


if __name__ == '__main__':
    print(DataAccessor.intents_filepath())
