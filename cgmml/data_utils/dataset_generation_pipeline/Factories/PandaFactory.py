from cgmzscore.src.main import z_score_wfh, z_score_lhfa
import pandas as pd

from LoggerPipe import LoggerPipe


class PandaFactory:

    @staticmethod
    def create_pose_data_frame(query_results_tmp_pose, column_names) -> pd.DataFrame:
        """
        Create a pose data DataFrame from query results.

        This function converts the provided query results and column names
        into a pandas DataFrame containing pose data.

        Args:
            query_results_tmp_pose: Query results containing pose data.
            column_names: Column names for the resulting DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the pose data.
        """
        return pd.DataFrame(query_results_tmp_pose, columns=list(map(lambda x: x.name, column_names)))

    @staticmethod
    def create_number_persons_data_frame(query_results, column_names) -> pd.DataFrame:
        """
        Create a number of persons data DataFrame from query results.

        This function converts the provided query results and column names
        into a pandas DataFrame containing the number of persons data.

        Args:
            query_results: Query results containing the number of persons data.
            column_names: Column names for the resulting DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the number of persons data.
        """
        df_no_of_person = pd.DataFrame(query_results, columns=list(map(lambda x: x.name, column_names)))
        return df_no_of_person

    @staticmethod
    def calculate_lhfa_zscore(rows: pd.DataFrame):
        """
        Calculate the length/height-for-age (LHFA) z-score for each row in the DataFrame.

        Args:
        rows (pd.DataFrame): DataFrame containing the columns 'sex', 'age', and 'height'.

        Returns:
        float: The calculated LHFA z-score.
        """
        sex = rows['sex']
        gender = 'F' if sex == 'female' else 'M'
        age = rows['age']
        height = rows['height']
        score = z_score_lhfa(age_in_days=age, sex=gender, height=height)
        return score

    @staticmethod
    def calculate_zscore(rows: pd.DataFrame):
        """
        Calculate the z-score for each row in the DataFrame.

        Args:
        rows (pd.DataFrame): DataFrame containing the columns 'sex', 'age', 'weight', and 'height'.

        Returns:
        float: The calculated z-score.
        """
        sex = rows['sex']
        gender = 'F' if sex == 'female' else 'M'
        age_in_days = float(rows['age'])
        weight = rows['weight']
        height = rows['height']
        score = z_score_wfh(weight=weight, age_in_days=age_in_days, sex=gender, height=height)
        return score

    @staticmethod
    def create_scans_data_frame(query_results, column_names: 'list[str]', logger: LoggerPipe) -> pd.DataFrame:
        """
        Create a DataFrame of scans data and filter it based on specific conditions.

        Parameters:
        -----------
        query_results : list of tuples
            Query results containing scans data.
        column_names : list
            List of column names for the DataFrame.

        Returns:
        --------
        df : pd.DataFrame
            Filtered DataFrame containing scans data.
        """
        dataframe = pd.DataFrame(query_results, columns=column_names)

        # Clean timestamp and convert to a formatted string
        dataframe['timestamp'] = dataframe['timestamp'].astype(str) \
            .apply(lambda ts: ts.replace(' ', '-').replace(':', '-').replace('.', '-'))

        logger.write(f"Total number of unique scans in fetched: {len(pd.unique(dataframe['scan_id']))})")

        # Insert fake manual measurements if they are not present
        for col, default_value in [('muac', 10), ('weight', 30), ('height', 90)]:
            if col not in dataframe.columns:
                dataframe.insert(4, col, default_value)

        logger.write(f"Shape with duplicates: {dataframe.shape}")
        dataframe = dataframe.drop_duplicates(subset=['scan_id', 'scan_step', 'timestamp', 'order_number'])
        logger.write(f"Shape without duplicates: {dataframe.shape}")

        def print_unique_counts(df, title):
            logger.write(title)
            logger.write(f"Unique persons: {len(df.person_id.unique())}")
            logger.write(f"Unique scan_ids: {len(df.scan_id.unique())}")
            logger.write(f"Unique artifacts: {len(df.file_path.unique())}")

        # Print unique counts before and after filtering by age
        print_unique_counts(dataframe, "Before filtering by age:")
        dataframe = dataframe.loc[(dataframe['age'] >= 365 / 2) & (dataframe['age'] <= 365 * 5)]
        print_unique_counts(dataframe, "After filtering by age:")

        return dataframe

    @staticmethod
    def create_standing_data_frame(query_results_tmp_standing, column_names) -> pd.DataFrame:
        """
        Create a DataFrame of standing data with an additional standing scores column.

        Parameters:
        -----------
        query_results_tmp_standing : list of tuples
            Query results containing standing data.
        column_names : list
            List of column names for the DataFrame.

        Returns:
        --------
        df_standing : pd.DataFrame
            DataFrame with standing data and a new standing scores column.
        """
        # Create DataFrame from query results and column names
        df_standing = pd.DataFrame(query_results_tmp_standing, columns=list(map(lambda x: x.name, column_names)))
        print("Total number of unique scans in standing:", len(pd.unique(df_standing['scan_id'])))

        # Add standing scores column using a lambda function and drop the 'data' column
        df_standing['standing'] = df_standing.apply(lambda rows: float(rows['data']['standing'][1:-1]), axis=1)
        df_standing = df_standing.drop(['data'], axis=1)

        return df_standing

    @staticmethod
    def diagnosis_on_lhfa(rows: pd.DataFrame):
        """
        Assign a diagnosis based on the length/height-for-age (LHFA) z-score in each row of the DataFrame.

        Args:
        rows (pd.DataFrame): DataFrame containing the 'lhfa' column.

        Returns:
        str: Diagnosis as 'Severely Stunted', 'Moderately Stunted', or 'Not Stunted'.
        """
        lhfa = rows['lhfa']

        if lhfa < -3:
            class_lhfa = 'Severely Stunted'
        elif -3 <= lhfa < -2:
            class_lhfa = 'Moderately Stunted'
        else:
            class_lhfa = 'Not Stunted'

        return class_lhfa

    @staticmethod
    def filter_rgbd_data(rows: pd.DataFrame):
        """
        This function filters the input DataFrame 'rows' based on the consistency of 'pose_score' and 'pose_result'
        between two formats: 'depth' and 'rgb'.

        First, it divides the input DataFrame into two DataFrames based on the 'format' column.

        It then traverses each row in the 'depth' DataFrame, identifies the corresponding row in the 'rgb' DataFrame
        (by 'scan_id' and 'order_number'), and verifies if 'pose_score' and 'pose_result' match in both rows.

        If they don't match, the indices of these rows are accumulated into a list of invalid rows.

        Post iteration, these invalid rows are discarded from both DataFrames.

        Lastly, it merges the remaining valid rows from both 'depth' and 'rgb' formats into one DataFrame and
        returns it.

        Parameters:
        rows (pd.DataFrame): The DataFrame to be processed.

        Returns:
        return_value (pd.DataFrame): The output DataFrame consisting of valid rows from both formats.
        """

        df_depth = rows[rows['format'] == 'depth']
        df_rgb = rows[rows['format'] == 'rgb']
        invalid_rows_depth = []
        invalid_rows_rgb = []
        for i in df_depth.index:
            scan_id = df_depth.loc[i, 'scan_id']
            order_number = df_depth.loc[i, 'order_number']

            # Find matching row in df_rgb
            matching_rows = df_rgb[(df_rgb['scan_id'] == scan_id) & (df_rgb['order_number'] == order_number)]

            if matching_rows.empty:
                # If no matching row, continue to next iteration
                continue

            for j in matching_rows.index:
                # If 'pose_score' and 'pose_result' don't match, mark as invalid
                if (df_depth.loc[i, 'pose_score'] != df_rgb.loc[j, 'pose_score']) or \
                      (df_depth.loc[i, 'pose_result'] != df_rgb.loc[j, 'pose_result']):
                    print(f"drop artifact id {df_depth.loc[i, 'artifact_id']} and {df_rgb.loc[j, 'artifact_id']}")
                    invalid_rows_depth.append(i)
                    invalid_rows_rgb.append(j)
        # Drop invalid rows from both dataframes
        df_depth.drop(invalid_rows_depth, inplace=True)
        df_rgb.drop(invalid_rows_rgb, inplace=True)
        # Concatenate the remaining rows into a single dataframe
        return_value = pd.concat([df_depth, df_rgb], ignore_index=True)

        return return_value

    @staticmethod
    def get_diagnosis(rows: pd.DataFrame):
        """
        Assign a diagnosis based on the z-score in each row of the DataFrame.

        Args:
        rows (pd.DataFrame): DataFrame containing the 'zscore' column.

        Returns:
        str: Diagnosis as 'SAM', 'MAM', or 'Healthy'.
        """
        zscore = rows['zscore']

        if zscore < -3:
            diagnosis = "SAM"
        elif -3 <= zscore < -2:
            diagnosis = "MAM"
        else:
            diagnosis = "Healthy"

        return diagnosis
