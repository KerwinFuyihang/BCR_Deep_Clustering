import numpy as np
import pandas as pd
from Bio import SeqIO
import os

class DataProcessor:
    def __init__(self, file_path: str):
        """
        Initialize the DataProcessor with a file path.

        Args:
            file_path (str): The path to the file containing data.
        """
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the file based on its extension.

        Returns:
            pd.DataFrame: Loaded data in a DataFrame.
        """
        file_extension = os.path.splitext(self.file_path)[1].lower()
        if file_extension == '.csv':
            return pd.read_csv(self.file_path)
        elif file_extension == '.tsv':
            return pd.read_csv(self.file_path, sep='\t')
        elif file_extension in ['.fasta', '.fa']:
            return pd.DataFrame({'sequence': self.read_fasta(self.file_path)})
        elif file_extension == '.txt':
            return pd.DataFrame({'line': self.read_txt(self.file_path)})
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def read_fasta(self, filepath: str) -> list:
        """
        Read sequences from a FASTA file.

        Args:
            filepath (str): Path to the FASTA file.

        Returns:
            list: List of sequences.
        """
        sequences = []
        for record in SeqIO.parse(filepath, "fasta"):
            sequences.append(str(record.seq))
        return sequences

    def read_txt(self, filepath: str) -> list:
        """
        Read lines from a text file.

        Args:
            filepath (str): Path to the text file.

        Returns:
            list: List of lines from the file.
        """
        with open(filepath, 'r') as f:
            return f.read().splitlines()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame by extracting relevant columns.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame with relevant columns.
        """
        if 'sequence_id' in df.columns:
            df = df[['sequence_id']]
        elif 'sequence' in df.columns:
            df = df[['sequence']]
        elif 'line' in df.columns:
            df = df[['line']]
        return df

    def truncate_sequence(self, sequences: list) -> list:
        """
        Truncate all sequences to the length of the shortest sequence.

        Args:
            sequences (list): List of sequences to truncate.

        Returns:
            list: List of truncated sequences.
        """
        target_length = min(len(seq) for seq in sequences)
        sequences = [seq[:target_length] for seq in sequences]
        return sequences

    def ensure_uppercase_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all sequences are in uppercase. If not, convert them to uppercase.

        Args:
            df (pd.DataFrame): DataFrame containing sequences.

        Returns:
            pd.DataFrame: DataFrame with sequences in uppercase.
        """
        if 'sequence' in df.columns:
            sequences = df['sequence'].tolist()
            # Check if all sequences are uppercase
            if not all(seq.isupper() for seq in sequences):
                # Convert to uppercase
                sequences = [seq.upper() for seq in sequences]
                df['sequence'] = sequences
        return df
