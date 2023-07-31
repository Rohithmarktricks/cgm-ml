

import csv
import psycopg2
from typing import List, Tuple


# Database connection
class DatabaseRepo:
    # Database connection
    """This class is used to connect to the database and get the data from the database.
    """
    def __init__(self, host: str, user: str, password: str):
        """ Constructor

        Args:
            host (str): _description_
            user (str): _description_
            password (str): _description_
        """
        conn = psycopg2.connect(host=host, database='cgm-ml', user=user, password=password, sslmode='require')
        self.__sql_cursor = conn.cursor()

    def __del__(self):
        """ Destructor
        """
        self.__sql_cursor.close()

    def get_number_persons_pose(self, workflow_id: str, num_artifacts: int = None) -> \
            Tuple[List[Tuple[str]], Tuple[psycopg2.extensions.Column]]:
        """
        Retrieve the number of persons using pose data from the database.

        This function queries the database to obtain the number of persons using
        pose for each artifact, as well as the corresponding artifact IDs and scan
        IDs, and returns the query results along with the column names.

        Args:
            num_artifacts (int, optional): Number of artifacts to fetch. If None, fetches all artifacts.

        Returns:
            Tuple[List[Tuple[str]], Tuple[psycopg2.extensions.Column]]: A tuple containing
            query results and column names.
        """
        limit_string = f"LIMIT {num_artifacts}" if num_artifacts is not None else ""
        sql_query_base_pose = f"""select ar.artifact_id, r.data->>'no of person using pose'
         as no_of_person, r.scan_id from artifact_result ar
        join "result" r on r.id = ar.result_id and r.result_workflow_id = '{workflow_id}' and r.data ?
         'no of person using pose'
        {limit_string}"""

        self.__sql_cursor.execute(sql_query_base_pose)
        query_results_tmp_pose: List[Tuple[str]] = self.__sql_cursor.fetchall()

        return query_results_tmp_pose, self.__sql_cursor.description

    def get_pose_result(self, workflow_id: str, num_artifacts: int = None) \
            -> Tuple[List[Tuple[str]], Tuple[psycopg2.extensions.Column]]:
        """
        Retrieves pose results from the database based on the specified number of artifacts.

        This function executes a SQL query to fetch pose results along with associated information
        from the artifact_result, result, and artifact tables. The query retrieves artifact_id, pose_score,
        pose_result, scan_id, ord, and format fields from the database. The number of artifacts to fetch
        can be limited using the num_artifacts parameter.

        Args:
            num_artifacts (int, optional): The maximum number of artifacts to fetch. If not provided,
                                            no limit is applied.

        Returns:
            Tuple[List[Tuple[str]], Tuple[psycopg2.extensions.Column]]: A tuple containing two elements:
                1. A list of tuples, each tuple representing a row of the query result.
                2. A tuple of psycopg2.extensions.Column objects, representing the column names
                  of the query result.
        """
        limit_string = f"LIMIT {num_artifacts}" if num_artifacts is not None else ""
        sql_query_base_pose = f""" select ar.artifact_id, r.data->>'Pose Scores'
          as pose_score, r.data->>'Pose Results' as pose_result,
          r.scan_id, a.ord, a.format from artifact_result ar
         INNER JOIN "result" r on r.id = ar.result_id
         INNER JOIN artifact a ON a.id =ar.artifact_id
         and r.result_workflow_id = '{workflow_id}' and r.data ? 'Pose Scores' and r.data ? 'Pose Results'
        {limit_string}"""

        self.__sql_cursor.execute(sql_query_base_pose)
        query_results_tmp_pose: List[Tuple[str]] = self.__sql_cursor.fetchall()
        column_names: Tuple[psycopg2.extensions.Column] = self.__sql_cursor.description

        return query_results_tmp_pose, column_names

    def get_scans(self, data_category: str, dataset_type: str, person_id: str, num_artifacts: int = None) \
            -> Tuple[List[Tuple[str]], List[str]]:
        """ Access SQL database to find all the scans/artifacts of interest
        We build our SQL query, so that we get all the required information for the ML dataset creation:
        - the artifacts (depthmap, RGB, pointcloud)
        - the targets (measured height, weight, and MUAC)
        MAGIC The ETL packet shows which tables are involved
        https://dev.azure.com/cgmorg/e5b67bad-b36b-4475-bdd7-0cf6875414df/_apis/git/repositories/465970a9-a8a5-4223-81c1-2d3f3bd4ab26/Items?path=%2F.attachments%2Fcgm-solution-architecture-etl-draft-ETL-samplling-71a42e64-72c4-4360-a741-1cfa24622dce.png&download=false&resolveLfs=true&%24format=octetStream&api-version=5.0-preview.1&sanitize=true&versionDescriptor.version=wikiMaster
        The query will produce one artifact per row.

        Returns:
            query_results, column_names
        """
        person_id_string = f"AND p.id = '{person_id}'" if person_id is not None else ""
        limit_string = f"LIMIT {num_artifacts}" if num_artifacts > -1 else ""
        sql_query_base = f"""
        SELECT f.file_path, f.created as timestamp,
            s.id as scan_id, s.scan_type_id as scan_step, s.version as scan_version,
            m.height, m.weight, m.muac,
            a.ord as order_number, a.format,
            a.file_id as artifact,
            a.id as artifact_id,
            di.model as device_model,
            p.id as person_id, p.age, p.sex
        FROM file f
        INNER JOIN artifact a ON f.id = a.file_id
        INNER JOIN scan s     ON s.id = a.scan_id
        INNER JOIN measure m  ON m.person_id = s.person_id
        INNER JOIN person p ON p.id = s.person_id
        INNER JOIN child_data_category cdc ON p.id = cdc.person_id
        INNER JOIN data_category dc ON dc.id = cdc.data_category_id
        INNER JOIN device_info di ON di.id = s.device_info_id
        WHERE dc.description = '{data_category}'
        AND di.model = 'HUAWEI VOG-L29'
        {person_id_string}
        """

        if dataset_type == 'depthmap':
            sql_query = f"""{sql_query_base}
            AND a.format IN ('depth', 'application/zip') {limit_string};"""
        elif dataset_type == 'rgbd':
            sql_query = f"""{sql_query_base}
            AND a.ord IS NOT NULL {limit_string};"""
        elif dataset_type == 'rgb':
            sql_query = f"""{sql_query_base}
            AND a.format IN('rgb') {limit_string};"""
        else:
            raise NameError(f'Unknown dataset type: {dataset_type}')

        self.__sql_cursor.execute(sql_query)

        # Get multiple query_result rows
        query_results: List[Tuple[str]] = self.__sql_cursor.fetchall()
        return query_results, list(map(lambda x: x.name, self.__sql_cursor.description))

    @staticmethod
    def get_scans_local(person_id: str, dataset_type: str, file_path: str) -> Tuple[List[Tuple[str]], List[str]]:
        """
        Retrieves scan data for a specific person from a CSV file and filters it based on
          the dataset type (RGB or depth).

        Args:
            person_id (str): The person ID to filter the data by.
            dataset_type (str): The type of dataset to filter the data by; should be either "rgb" or "depth".
            file_path (str): The path to the CSV file containing the scan data.

        Returns:
            tuple: A tuple containing two elements:
                - data (list of tuples): A list of tuples representing the filtered scan data. Each tuple contains the
                values of the required columns in the same order as they appear in the required_columns list.
                - required_columns (list of str): A list of the required column names.
        """
        data = []
        required_columns = ['file_path', 'timestamp', 'scan_id',
                            'scan_step', 'scan_version', 'height',
                            'weight', 'muac', 'order_number',
                            'format', 'artifact', 'artifact_id',
                            'device_model', 'person_id', 'age', 'sex']
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['person_id'] != person_id:
                    continue
                new_row = {key: row[key] for key in required_columns}
                if (dataset_type == "rgb" and new_row["format"] != "rgb"):
                    continue
                if (dataset_type == "depthmap" and new_row["format"] != "depth"):
                    continue
                new_row['scan_step'] = int(new_row['scan_step'])
                new_row['age'] = int(new_row['age'])
                new_row['height'] = float(new_row['height'])
                new_row['weight'] = float(new_row['weight'])
                new_row['muac'] = float(new_row['muac'])
                new_row['order_number'] = int(float(new_row['order_number']))
                row_tuple = tuple(new_row[key] for key in required_columns)
                data.append(row_tuple)

        return data, required_columns
