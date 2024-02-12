import argparse
import networkx as nx
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from utils import *
import random
import rdflib
from rdflib.namespace import RDF, RDFS

   
class OntologyPreprocessor:
    """
    Processes ontology data to generate and save subsumption triples, and to create a pickle file of pairs/triples and labels.
    """
    def __init__(self):
        self.case_name = None
        self.ontology = None
        self.graph = None
        self.triples = pd.DataFrame(columns=['head', 'relation', 'tail'])

    def convert_to_graph(self):
        """
        Varies per ontology, defined in subclass
        """
        raise NotImplementedError

    def generate_subsumption_triples(self) -> pd.DataFrame:
        """
        Generates subsumption triples from the ontology graph, considering 'narrowmatch' and 'broadmatch' relations.

        :return: A DataFrame of generated triples.
        """
        triples = []
        for edge in self.graph.edges():
            parent, child = edge
            if not parent.lower() == child.lower():
                triples.append((parent.lower(), 'narrowmatch', child.lower()))
                triples.append((child.lower(), 'broadmatch', parent.lower()))

        df_triples = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])

        self.triples = pd.concat([self.triples, df_triples], ignore_index=True)

    def save_triples(self):
        """
        Saves the generated triples to a CSV file, naming it according to the case_name attribute.
        """
        save_processed_df_to_csv(
            self.triples, f"{self.case_name}_triples", "triples"
        )


    def get_triples(self):
        """
        Returns the DataFrame of generated triples.

        :return: The triples DataFrame.
        """
        return self.triples
    
    def remove_duplicate_pairs(self):
        """
        Removes duplicate pairs from the triples DataFrame, prioritizing 'exactmatch' relations.
        Prints the number of removed duplicate pairs.
        """
        triples = self.triples
        len_before = len(triples)
                
        triples['is_exactmatch'] = triples['relation'] == 'exactmatch'
        triples = triples.sort_values(by='is_exactmatch', ascending=False)
        triples = triples.drop('is_exactmatch', axis=1)

        triples = triples.drop_duplicates(['head','tail'], keep='first')
        len_after = len(triples)
        len_diff = len_before - len_after

        self.triples = triples

        print(f"Removed {len_diff} duplicate pairs")

    def print_triples_statistics(self) -> None:
        """
        Prints statistics of the generated triples, including the total number of triples and counts of each relation type.
        """
        print("-------------- Triple statistics ------------------")
        print(f'Number of triples: {len(self.triples)}')

        relation_counts = self.triples['relation'].value_counts().to_dict()
        print("Relation Counts Top 20:")
        counter = 0
        for relation, count in relation_counts.items():
            print(f"{relation}: {counter}")
            if counter > 18:
                break



class StromaExternalDataPreprocessor(OntologyPreprocessor):
    """
    A preprocessor for external data related to the STROMA (web) use cases, extending the OntologyPreprocessor class.
    It initializes with data specific to a use case name, reads raw CSV files, and preprocesses them.
    """
    def __init__(self, case_name):
        """
        Initializes the StromaExternalDataPreprocessor with a case name, loads triples from a CSV file,
        and performs initial preprocessing steps.

        :param case_name: The name of the case to be processed, used to load specific raw data files.
        """
        self.case_name = case_name
        self.triples = pd.DataFrame(columns=['head', 'relation', 'tail'])

        if self.case_name == 'wikidata':
            self.triples = read_raw_csv(
                f"stroma_train_data_{self.case_name}",
                delimiter=',',
                encoding='utf-8'
            )
        else:
            self.triples = read_raw_csv(f"stroma_train_data_{self.case_name}")

        # there were some ; in the strings so they are replaced by ',' to counter delim errors
        self.triples = self.triples.replace({';': ','}, regex=True)

        str_cols = self.triples.select_dtypes(include=['object'])
        self.triples[str_cols.columns] = str_cols.apply(lambda x: x.str.lower())


    def convert_ints_to_labels(self):
        """
        Converts integer codes in the triples to their corresponding labels using a predefined dictionary.
        This method is specific to the preprocessing of STROMA data where relation types are initially represented as integers.
        """
        self.triples = self.triples.replace(STROMA_MODEL_LABEL_DICT)



class ONETOntologyPreprocessor(OntologyPreprocessor):
    """
    Processes and converts O*NET and SOC data into a graph representation and converts it to triples
    """
    def __init__(self, case_name):
        super().__init__()
        self.case_name = case_name

    def read_SOC(self):
        """
        Reads and processes SOC data, standardizing the format and preparing it for conversion into a graph.
        """
        soc = read_raw_csv("SOC")
        soc = soc.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        soc_list = []
        for row_idx in range(len(soc)):
            for column_idx in range(4):
                if isinstance(soc.iloc[row_idx,column_idx], str):
                    soc_list.append((soc.iloc[row_idx,column_idx], soc.iloc[row_idx,4]))

        df_soc = pd.DataFrame(soc_list, columns=['code','label'])

        df_soc['code'] = df_soc['code'].apply(lambda x: x[:6] + '-' + x[6:] + "-00")
        df_soc['code'] = df_soc['code'].apply(lambda x: x[:4] + '-' + x[4:])
 
        self.soc = df_soc

    def read_ONET(self):
        """
        Reads and processes O*NET data, standardizing the format and preparing it for conversion into a graph.
        """
        df_onet = read_raw_csv("ONET")
        df_onet = df_onet.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        df_onet = df_onet[["ï»¿O*NET-SOC 2019 Code", "O*NET-SOC 2019 Title"]]
        df_onet.columns = ['code','label']

        df_onet['code'] = df_onet['code'].apply(lambda x: x.replace('.','-'))
        df_onet['code'] = df_onet['code'].apply(lambda x: x[:6] + '-' + x[6:])
        df_onet['code'] = df_onet['code'].apply(lambda x: x[:4] + '-' + x[4:])

        self.onet = df_onet
    
    def convert_to_graph(self):
        """
        Converts the processed O*NET and SOC data into a graph structure, establishing hierarchical relationships.
        """
        G= nx.DiGraph()
        for code in self.onet['code']:
            # split the code into its components
            parts = code.split('-')

            # add edges for each parent-child pair
            first = parts[0] + '-0-00-0-00'
            second = parts[0] + '-' + parts[1] + '-00-0-00'
            third = parts[0] + '-' + parts[1] + '-' + parts[2] + '-0-00'
            fourth = parts[0] + '-' + parts[1] + '-' + parts[2] + '-' + parts[3] + '-00'
            fifth = parts[0] + '-' + parts[1] + '-' + parts[2] + '-' + parts[3] + '-' + parts[4]

            G.add_edge(first, second)
            G.add_edge(second, third)
            G.add_edge(third, fourth)
            G.add_edge(fourth, fifth)


        soc_label_dict = self.soc.set_index('code')['label'].to_dict()
        onet_label_dict = self.onet.set_index('code')['label'].to_dict()
        for node in G.nodes:
            if len(node) == 2:
                G = nx.relabel_nodes(G, {node: node})
            if len(node) == 6:
                G = nx.relabel_nodes(G, {node: node})
            if len(node) == 8:
                G = nx.relabel_nodes(G, {node: node})

        H = nx.relabel_nodes(G, soc_label_dict)
        H2 = nx.relabel_nodes(H, onet_label_dict)

        self.graph = H2
        
    def fix_inconsistencies(self):
        """
        Fixes inconsistencies in the triples data, specifically handling known misalignments and labeling issues.
        """
        triples = self.triples

        replace_dict = {"51-5-00-0-00":"Printing Workers",
                        "15-1-00-0-00":"Computer Occupations",
                        "29-1-22-0-00":"Physicians",
                        "31-1-00-0-00":"Home Health and Personal Care Aides; and Nursing Assistants, Orderlies, and Psychiatric Aides"}
        triples = triples.replace(replace_dict)

        self.triples = triples

class ESCOOntologyPreprocessor(OntologyPreprocessor):
    """
    Processes ESCO ontology data into a graph and convert to triples
    """
    def __init__(self, case_name):
        super().__init__()
        self.case_name = case_name
        self.ontology = read_raw_csv("ESCO_engels", encoding="utf-8")
        self.ontology = self.ontology[['parentLabel','childLabel']]
        self.ontology = self.ontology.rename({"parentLabel":'head','childLabel':'tail'}, axis=1)
        self.ontology = self.ontology.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    def generate_exactmatch_ESCO(self) -> None:
        """
        Generates exact match triples from raw ESCO data, particularly focusing on alternative labels.
        """
        raw_altLabels = read_raw_csv('raw_translated_synonyms')
        raw_altLabels = raw_altLabels.drop_duplicates('preferredLabel', keep='first')
        raw_altLabels = raw_altLabels.dropna()

        altlabel_exactmatches = raw_altLabels[['preferredLabel','altLabels']].rename({'preferredLabel':'head','altLabels':'tail'}, axis=1)
        altlabel_exactmatches['head'] = altlabel_exactmatches['head'].str.lower()
        altlabel_exactmatches['tail'] = altlabel_exactmatches['tail'].str.lower()
        altlabel_exactmatches['relation'] = 'exactmatch'

        self.triples = pd.concat([self.triples, altlabel_exactmatches], ignore_index=True) 

    def convert_to_graph(self):
        """
        Converts the processed ESCO data into a graph structure, capturing the hierarchical relationships.
        """
        G = nx.DiGraph()

        for index, row in self.ontology.iterrows():
            G.add_edge(row['head'], row['tail'])

        self.graph = G
    
    
class CNLOntologyPreprocessor(OntologyPreprocessor):
    """
    Processes CompetentNL ontology data, converting it into a graph representation and into triples
    """
    def __init__(self, case_name):
        super().__init__()
        self.case_name = case_name
        self.ontology = read_raw_csv("CNL_engels", encoding='cp1252')
        self.ontology = self.ontology.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    def preprocess_raw_ontology(self) -> pd.DataFrame:
        """
        Preprocesses the raw CNL data, standardizing formats and preparing for graph conversion.
        """
        self.ontology = self.ontology.astype(str)
        columns_and_lengths = [
            ('code 5e laag', 5),
            ('isco code UG', 4),
            ('isco code MiG', 3),
            ('isco code sub MG', 2)
        ]
        for column, length in columns_and_lengths:
            self.ontology[column] = left_fill_str_with_zeroes(
                self.ontology[column], length)
            
    def convert_to_graph(self) -> nx.DiGraph:
        """
        Converts the processed CNL data into a graph structure
        """
        label_dict = create_CNL_occupation_label_dict(self.ontology)
        self.ontology['fullPath'] = self.ontology[
            ['isco code MG', 'isco code sub MG', 'isco code MiG',
             'isco code UG', 'code 5e laag', 'BEROEPS_CODE']
        ].agg('-'.join, axis=1)

        G = nx.DiGraph()

        for path in self.ontology['fullPath']:
            nodes = path.split('-')
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i + 1])

        nx.set_node_attributes(G, label_dict, 'label')

        named_G = nx.DiGraph()

        for node in G.nodes():
            node_label = G.nodes[node]['label']
            named_G.add_node(node_label, id=node)

        for edge in G.edges():
            parent, child = edge
            parent_label = G.nodes[parent]['label']
            child_label = G.nodes[child]['label']
            named_G.add_edge(parent_label, child_label)

        self_loops = list(nx.selfloop_edges(named_G))
        named_G.remove_edges_from(self_loops)

        self.graph = named_G

    def remove_plural_subsumptions(self):
        """
        Removes plural subsumptions from the graph to avoid redundancy
        """
        len_before = len(self.triples)

        mask = ~((self.triples['head'] == self.triples['tail'] + 's') | (self.triples['tail'] == self.triples['head'] + 's'))
        
        self.triples = self.triples[mask]
        len_after = len(self.triples)
        
        diff = len_before - len_after
        print(f'Number of plural subsumptions dropped: {diff}')


class STROMAOntologyPreprocessor(OntologyPreprocessor):
    """
    Processes STROMA ontology data from RDF format into a DataFrame structure suitable for further processing and analysis. This class is tailored for handling RDF data specific to the STROMA project ontologies.
    """
    def __init__(self, case_name):
        super().__init__()
        self.graph = read_raw_rdf(f"stroma_{case_name}", "graph")
        self.case_name = f"stroma_{case_name}"

    def RDF_to_df(self):
        """
        Converts RDF data from the loaded graph into a pandas DataFrame, extracting necessary information based on predefined queries.
        """
        if self.case_name == "stroma_g7_source":
            return None

        standard_q = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>

                SELECT ?class ?label ?subClassOf
                WHERE {
                ?class rdf:type owl:Class .
                ?class rdfs:label ?label .
                ?class rdfs:subClassOf ?subClassOf .
                }
            """
        no_label_q = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT ?class ?subClassOf
            WHERE {
            ?class rdf:type owl:Class .
            ?class rdfs:subClassOf ?subClassOf .
            }
            """

        queries ={
            "g1":standard_q,
            "g2":standard_q,
            "g3":standard_q,
            "g4":standard_q,
            "g5":no_label_q,
            "g6":no_label_q,
            "g7":no_label_q
            }
        
        q_key = self.case_name[7:9]
        qres = self.graph.query(queries[q_key])

        df = pd.DataFrame(qres)

        if len(df.columns) > 2:
            df = df.drop(0, axis=1)

        df.columns = ['head','tail']
        df['relation'] = "broadmatch"

        for column in ['head', 'tail']:
            links = df[column]
            entities = [link.rsplit('/', 1)[-1] for link in links]
            entities = [x.replace("_", " ") for x in entities]
            df[column] = entities

        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        self.triples = df

    def get_inverse_triples(self):
        """
        Generates inverse triples from the existing triples DataFrame, switching heads and tails and changing the relation to 'narrowmatch'.
        """
        inverse_df = pd.DataFrame()
        inverse_df['head'] = self.triples['tail']
        inverse_df['tail'] = self.triples['head']
        inverse_df['relation'] = 'narrowmatch'

        self.triples = pd.concat([self.triples, inverse_df])


class BIOOntologyPreprocessor(OntologyPreprocessor):
    """
    Processes biological ontology data, particularly focusing on synonyms and subsumption relationships. This class is designed to handle data specific to biological ontologies.
    """
    def __init__(self, case_name):
        super().__init__()
        self.triples = read_raw_csv(f"bio_triples/{case_name}_triples_synonyms")
        self.triples = self.triples.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        self.case_name = case_name

        print('rel counts:', self.triples['relation'].value_counts())


    def generate_subsumption_triples_bio(self) -> DataFrame:
        """
        Generates subsumption triples (broadmatch and narrowmatch) from the existing exact match triples in the dataset.
        """
        exactmatches = self.triples[self.triples['relation']=='exactmatch'].copy()
        broadmatches = self.triples[self.triples['relation']=='broadmatch'].copy()
        narrowmatches = pd.DataFrame()
        narrowmatches[['tail','head']] = broadmatches[['head','tail']]
        narrowmatches['relation'] = 'narrowmatch'

        self.triples = pd.concat([exactmatches, broadmatches, narrowmatches])

        print('rel counts:', self.triples['relation'].value_counts())

    def subset_exactmatches(self):
        """
        Subsets exact match triples to only include the top 1 exact match per head, reducing redundancy in the dataset.
        """
        exactmatches = self.triples[self.triples['relation'] == 'exactmatch'].copy()
        grouped_top1 = exactmatches.groupby('head').head(1).reset_index(drop=True)

        all_triples = self.triples.copy()
        bm_nm_triples = all_triples[(all_triples['relation'] == 'broadmatch') | (all_triples['relation'] == 'narrowmatch')]

        all_triples_new = pd.concat([grouped_top1, bm_nm_triples])
        self.triples = all_triples_new

class MappingPreprocessor:
    """
    A superclass for mapping preprocessors, providing a common interface and implementations for 
    processing, saving, and deduplicating mapping data.
    """
    def __init__(self):
        self.mapping = None
        self.case_name = None

    def process_raw_mapping(self):
        """
        Varies per mapping, will be specified in subclasses
        """
        raise NotImplementedError

    def save_mapping_triples(self):
        """
        Saves the processed mapping data to a CSV file, naming it according to the case_name attribute.
        """
        save_processed_df_to_csv(
            self.mapping, f"{self.case_name}_triples", "triples"
        )

    def drop_duplicate_rows(self):
        """
        Removes duplicate rows from the mapping data, prioritizing rows with a specific 'relation' value 
        ('exactmatch' by default) and then removing any remaining duplicates based on 'head' and 'tail' columns.
        """
        triples = self.mapping.copy()

        preferred_value = 'exactmatch'
        triples_sorted = triples.sort_values(by='relation', key=lambda x: x == preferred_value, ascending=False)
        triples_deduplicated = triples_sorted.drop_duplicates(subset=['head', 'tail'], keep='first').sort_index()

        self.mapping = triples_deduplicated


class ESCOCNLMappingPreprocessor(MappingPreprocessor):
    """
    Processes and prepares mappings between ESCO and CNL classifications, converting raw mapping data into a structured format suitable for further analysis or model training.
    """
    def __init__(self):
        super().__init__()
        self.mapping = read_raw_csv("ESCO-CNL_engels_compleet", encoding='cp1252')
        self.mapping = self.mapping.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        self.case_name = "ESCO-CNL"

    def process_raw_mapping(self):
        """
        Processes the raw ESCO-CNL mapping data, renaming columns and filtering rows to ensure data integrity.
        """
        rename_dict = {
            "Classification_2_PrefLabel": "tail",
            'Classification_1_PrefLabel': 'head',
            'Mapping_relation': 'relation'
        }

        self.mapping = (
            self.mapping[['Classification_2_PrefLabel', 'Classification_1_PrefLabel', 'Mapping_relation']]
            .rename(columns=rename_dict)
            .dropna()
            .loc[lambda df: ~(df == 'nan').any(axis=1)]
        )

        self.mapping['relation'] = self.mapping['relation'].str[5:]

    def remove_closematch_rows(self):
        """
        Removes mappings labeled as 'closematch' to focus on more definitive relation types.
        """
        self.mapping = self.mapping[self.mapping['relation'] != 'closematch']

class ESCOONETMappingPreprocessor(MappingPreprocessor):
    """
    Processes mappings between ESCO and O*NET, standardizing and preparing the data for use in mapping tasks.
    """
    def __init__(self):
        super().__init__()
        self.mapping = read_raw_csv('ESCO-ONET')
        self.mapping = self.mapping.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        self.case_name = "ESCO-ONET"

    def process_raw_mapping(self):
        """
        Standardizes the raw mapping data between ESCO and O*NET, setting up the structure for analysis or model input.
        """
        self.mapping = self.mapping[["O*NET Title", "ESCO or ISCO Title","Type of Match"]]
        self.mapping.columns = ['tail','head','relation']


class ESCOBLMMappingPreprocessor(MappingPreprocessor):
    """
    Processes mappings between ESCO and BLM (Berufliche Landkarten), preparing the data for tasks such as relation prediction or ontology alignment.
    """
    def __init__(self):
        super().__init__()
        self.mapping = read_raw_csv("ESCO-BLM_mapping")
        self.mapping = self.mapping.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        self.case_name = "ESCO-BLM"

    def process_raw_mapping(self) -> pd.DataFrame:
        """
        Processes the raw mapping data between ESCO and BLM, standardizing and deduplicating entries.
        """
        rename_dict = {
            'Classification 1 PrefLabel': "head",
            'Classification 2 PrefLabel NL': "tail",
            'Mapping relation': "relation"
        }

        self.mapping = (
            self.mapping[
                ['Classification 1 PrefLabel', 'Classification 2 PrefLabel NL', 'Mapping relation']
            ]
            .rename(columns=rename_dict)
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        self.mapping['relation'] = self.mapping['relation'].str[5:]


    def remove_closematch_rows(self):
        """
        Removes mappings labeled as 'closematch' to focus on more definitive relation types.
        """
        self.mapping = self.mapping[self.mapping['relation'] != 'closematch']

    def add_inverse_subsumptions(self):
        """
        Generates inverse subsumption relations to enrich the dataset with additional semantic information.
        """
        inverse_triples = self.mapping[
            self.mapping['relation'].isin(['narrowmatch', 'broadmatch'])
        ].copy()
        
        inverse_triples[['head', 'tail']] = inverse_triples[['tail', 'head']]
        inverse_triples['relation'] = inverse_triples['relation'].replace(
            {'narrowmatch': 'broadmatch', 'broadmatch': 'narrowmatch'}
        )
        
        self.mapping = pd.concat([self.mapping, inverse_triples], ignore_index=True)


class RDFMappingPreprocessor(MappingPreprocessor):
    """
    Specializes in processing RDF mapping data, converting RDF structures into a tabular format for tasks like entity alignment or relation prediction.
    """
    def __init__(self, dataset_id):
        super().__init__()
        self.tree = read_raw_rdf(f"stroma_{dataset_id}_reference", "tree")
        self.case_name = f"stroma_{dataset_id}_reference"

    def RDF_to_df(self):
        """
        Converts RDF data into a DataFrame, standardizing entity and relation representations for easier manipulation.
        """
        tree = self.tree

        namespaces = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                    'align': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'}

        entity1_list = []
        entity2_list = []
        relation_list = []

        for cell in tree.xpath('//align:map/align:Cell', namespaces=namespaces):
            entity1 = cell.xpath('./align:entity1/@rdf:resource', namespaces=namespaces)
            entity2 = cell.xpath('./align:entity2/@rdf:resource', namespaces=namespaces)
            relation = cell.xpath('./align:relation/text()', namespaces=namespaces)
            
            entity1_list.append(entity1[0] if entity1 else None)
            entity2_list.append(entity2[0] if entity2 else None)
            relation_list.append(relation[0] if relation else None)

        df = pd.DataFrame({
            'head': entity1_list,
            'tail': entity2_list,
            'relation': relation_list,
        })
        
        df['relation'] = df['relation'].replace({'=':'exactmatch','>':'narrowmatch','<':'broadmatch'})

        for column in ['head', 'tail']:
            links = df[column]
            entities = [link.rsplit('/', 1)[-1] for link in links]
            entities = [x.replace("_", " ") for x in entities]
            df[column] = entities

        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        self.mapping = df

    def generate_broadmatch_triples(self):
        """
        From the existing 'narrowmatch' relations, generates 'broadmatch' relations to provide a more comprehensive mapping dataset.
        """
        narrowmatch_triples = self.mapping[self.mapping['relation']=='narrowmatch']
        broadmatch_triples = pd.DataFrame()
        broadmatch_triples[['head','tail']] = narrowmatch_triples[['tail','head']]
        broadmatch_triples['relation'] = 'broadmatch'

        self.mapping = pd.concat([self.mapping, broadmatch_triples])




class BioMappingPreprocessor(MappingPreprocessor):
    """
    Dedicated to processing biological entity mappings, incorporating various types of semantic relations such as equivalence or subsumption.
    """
    def __init__(self, case_name):
        super().__init__()
        self.case_name = case_name
        test_case_name = self.case_name.split('-')
        self.source_ontology = test_case_name[0]
        self.target_ontology = test_case_name[1]

        equivalence_mappings = read_raw_csv(f"{case_name}/refs_equiv/test.cands", delimiter='\t',extension='.tsv')
        equivalence_mappings['relation'] = 'exactmatch'
        subsumption_mappings = read_raw_csv(f"{case_name}/refs_subs/test.cands", delimiter='\t',extension='.tsv')
        subsumption_mappings['relation'] = 'broadmatch'
        self.mapping = pd.concat([equivalence_mappings, subsumption_mappings])
        self.mapping = self.mapping.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        for col in  self.mapping.columns:
             self.mapping[col] =  self.mapping[col].map(str)

    def process_raw_mapping(self):
        """
        Processes the raw biological mapping data, structuring it for tasks such as entity alignment in the biological domain.
        """
        self.mapping = self.mapping[['SrcEntity','TgtEntity','relation']]
        self.mapping.columns = ['head','tail','relation']

    def replace_urls_with_labels(self):
        """
        Replaces URLs or identifiers with human-readable labels to facilitate understanding and analysis of the mappings.
        """
        def get_label_mapping(ontology):
            g = rdflib.Graph()
            g.parse(f'../data/raw/{self.case_name}/{ontology}.owl', format='xml')
                        
            query = """
            SELECT DISTINCT ?class ?label
            WHERE {
                ?class a owl:Class.
                OPTIONAL {?class rdfs:label ?label.}
            }
            """

            results = g.query(query)

            data = []
            for row in results:
                uri = str(row['class'])
                label = str(row['label']) if row['label'] else None
                data.append({'uri': uri.strip(), 'label': label})

            df = pd.concat([pd.DataFrame([row]) for row in data], ignore_index=True).dropna()

            return df 

        source_mapping = get_label_mapping(self.source_ontology)
        source_mapping['uri'] = source_mapping['uri'].str.strip().str.lower()
        self.mapping['head'] = self.mapping['head'].str.strip().str.lower()
        self.mapping['head'] = self.mapping['head'].map(source_mapping.set_index('uri')['label'])
        self.mapping['head'] = self.mapping['head'].str.lower()

        target_mapping = get_label_mapping(self.target_ontology)
        target_mapping['uri'] = target_mapping['uri'].str.strip().str.lower()
        self.mapping['tail'] = self.mapping['tail'].str.strip().str.lower()
        self.mapping['tail'] = self.mapping['tail'].map(target_mapping.set_index('uri')['label'])
        self.mapping['tail'] = self.mapping['tail'].str.lower()

    def generate_narrowmatch_triples(self):
        """
        Generates 'narrowmatch' relations from the 'broadmatch' relations to enrich the dataset with inverse semantic information.
        """
        broadmatch_triples = self.mapping[self.mapping['relation']=='broadmatch']
        narrowmatch_triples = pd.DataFrame()
        narrowmatch_triples[['head','tail']] = broadmatch_triples[['tail','head']]
        narrowmatch_triples['relation'] = 'narrowmatch'

        self.mapping = pd.concat([self.mapping, narrowmatch_triples])

    def subsample_if_necessary(self):
        """
        Subsamples the mapping data if the total number of mappings exceeds a predefined threshold, ensuring manageable dataset sizes.
        """
        if len(self.mapping) > 5000:
            self.mapping = self.mapping.groupby('relation', group_keys=False).apply(lambda x: x.sample(int(np.rint(5000 * len(x) / len(self.mapping))))).sample(frac=1).reset_index(drop=True)
            

class DataSetConverter:
    """
    Provides functionality to convert processed ontologies into training and testing datasets,
    including adding positive labels, shuffling, generating inverse subsumptions, and removing duplicates.
    Intended as a base class for specific dataset conversion implementations such as TrainSetConverter and TestSetConverter.
    """
    def __init__(self, set_name, case_name):
        self.set_name = set_name
        self.case_name = case_name
        self.balanced_bool = False 
        self.triples = None
        self.validation_triples = None
        self.test_triples = None
        self.test_case_name = None

    def add_pos_label_to_df(self, set_name):
        """
        Adds a positive label to the dataset specified by set_name.
        """
        if set_name == "train":
            self.triples['label'] = 1
        elif set_name == "validation":
            self.validation_triples['label'] = 1
        elif set_name == "test":
            self.test_triples['label'] = 1
        else:
            sys.exit("something is wrong in add_pos_label_to_dict")

    def shuffle_triples_random(self):
        """
        Randomly shuffles the triples in the main dataset.
        """
        self.triples = self.triples.sample(frac=1, random_state=234).reset_index(drop=True)

    def add_inverse_subsumptions(self):
        """
        Adds inverse subsumption relations to the dataset to enhance learning.
        """
        inverse_triples = self.triples[
            self.triples['relation'].isin(['narrowmatch', 'broadmatch'])
        ].copy()
        
        inverse_triples[['head', 'tail']] = inverse_triples[['tail', 'head']]
        inverse_triples['relation'] = inverse_triples['relation'].replace(
            {'narrowmatch': 'broadmatch', 'broadmatch': 'narrowmatch'}
        )
        
        self.triples = pd.concat([self.triples, inverse_triples], ignore_index=True)

    def drop_rows_with_na(self, set_name):
        """
        Drops rows with any NA values in the specified dataset.
        """
        set_types = {"train": self.triples,
                        "test": self.test_triples,
                        "validation": self.validation_triples}
        triples = set_types[set_name]

        mask = triples[['head', 'relation', 'tail']].isna().any(axis=1)
        rows_without_na = triples[~mask]
        print(f"{len(triples[mask])} rows contained NA's and were dropped")

        if set_name == "train":
            self.triples = rows_without_na
        elif set_name == "validation":
            self.validation_triples = rows_without_na
        elif set_name == "test":
            self.test_triples = rows_without_na

    def strip_rows(self, set_name) -> None:
        """
        Strips whitespace from all string values in the specified dataset.
        """
        set_types = {"train": self.triples,
                        "test": self.test_triples,
                        "validation": self.validation_triples}
        strip_triples = set_types[set_name]

        strip_triples = strip_triples.map(lambda x: x.strip() if isinstance(x, str) else x)

        if set_name == "train":
            self.triples = strip_triples
        elif set_name == "test":
            self.test_triples = strip_triples
        elif set_name == "validation":
            self.validation_triples = strip_triples

    def generate_all_negatives(self, set_name):
        """
        Generates all possible negative examples for the specified dataset by flipping relations.
        """
        set_types = {"train": self.triples,
                        "test": self.test_triples,
                        "validation": self.validation_triples}
        true_triples = set_types[set_name]

        if hasattr(self, 'case_name'):
            case_name_str = self.case_name
        elif hasattr(self, 'case_name'):
            case_name_str = self.case_name

        elif "CNL" in case_name_str or "ESCO" in case_name_str:
            relation_list = ESCO_CNL_RELATIONS
        elif "stroma" in case_name_str:
            relation_list = STROMA_DATA_RELATIONS
        else:
            sys.exit(
                "Could not select relation list, check function "
                "'generate_all_negatives' if still up to date with the used datasets"
            )

        false_triples = []
        for _, row in true_triples.iterrows():
            for relation in relation_list:
                if relation != row['relation']:
                    false_triples.append([row['head'], relation, row['tail'], 0])

        false_triples_df = pd.DataFrame(
            false_triples, columns=['head', 'relation', 'tail', 'label']
        )

        true_triples = pd.concat([true_triples, false_triples_df], ignore_index=True)
    
        if set_name == "train":
            self.triples = true_triples
        elif set_name == "test":
            self.test_triples = true_triples
        elif set_name == "validation":
            self.validation_triples = true_triples

    def drop_duplicate_rows(self, set_name):
        """
        Drops duplicate rows based on 'head' and 'tail' columns in the specified dataset.
        """
        set_types = {"train": self.triples,
                    "validation": self.validation_triples,
                    "test": self.test_triples}
        triples = set_types[set_name]

        preferred_value = 'exactmatch'
        triples_sorted = triples.sort_values(by='relation', key=lambda x: x == preferred_value, ascending=False)

        if set_name == "train":
            self.triples = triples_sorted.drop_duplicates(subset=['head', 'tail'], keep='first').sort_index()
            print(f"Number of duplicates dropped in set converter: {len(triples)-len(self.triples)}")

        elif set_name == "validation":
            self.validation_triples = triples_sorted.drop_duplicates(subset=['head', 'tail'], keep='first').sort_index()
            print(f"Number of duplicates dropped in set converter: {len(triples)-len(self.validation_triples)}")

        elif set_name == "test":
            self.test_triples = triples_sorted.drop_duplicates(subset=['head', 'tail'], keep='first').sort_index()
            print(f"Number of duplicates dropped in set converter: {len(triples)-len(self.test_triples)}")

    def save_to_triples_TC(self, set_name, anchor_folder):
        """
        Saves the dataset to files suitable for Triple Classification (TC), including pairs/triples and labels.
        """
        set_types = {"train": self.triples,
                     "validation": self.validation_triples,
                     "test": self.test_triples}
        triples = set_types[set_name]

        if (set_name == "test") and (anchor_folder != "default"):
            self.case_name = self.test_case_name
        
        triples_list = list(
            triples[['head', 'relation', 'tail']].itertuples(
                index=False, name=None)
        )
        labels = list(triples['label'])
        le = LabelEncoder()
        int_relations = le.fit_transform(triples['relation'])
        
        if self.balanced_bool == True:
            balanced_str = "_BA"
        elif self.balanced_bool == False:
            balanced_str = "_UB"
        elif self.balanced_bool == None:
            balanced_str = ""
        else:
            sys.exit("Something is wrong with you entry for balancing the trainset, check the save_to_triples_TC function")

        if set_name in ['validation','test']:
            balanced_str = ""

        save_processed_df_to_csv(
            triples, f"TC_{set_name}_{self.case_name}{balanced_str}", f"{set_name}/{anchor_folder}/{self.case_name}")
        save_processed_var_to_pickle(
            triples_list, f'TC_{set_name}_triples_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")
        save_processed_var_to_pickle(
            labels, f'TC_{set_name}_labels_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")

        if set_name == "train":
            save_processed_var_to_pickle(
                int_relations, f'TC_{set_name}_relations_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")

    def save_to_pairs_RP(self, set_name, anchor_folder):
        """
        Saves the dataset to files suitable for Relation Prediction (RP), including pairs and integer-encoded labels.
        """
        set_types = {"train": self.triples,
                     "validation": self.validation_triples,
                     "test": self.test_triples}
        triples = set_types[set_name]

        if (set_name == "test") and (anchor_folder != "default"):
            self.case_name = self.test_case_name

        pairs_list = list(
            triples[['head', 'tail']].itertuples(index=False, name=None)
        )
        le = LabelEncoder()
        integer_labels = le.fit_transform(triples['relation'])

        if self.balanced_bool == True:
            balanced_str = "_BA"
        elif self.balanced_bool == False:
            balanced_str = "_UB"

        if set_name in ['validation','test']:
            balanced_str = ""

        save_processed_df_to_csv(
            triples, f"RP_{set_name}_{self.case_name}{balanced_str}", f"{set_name}/{anchor_folder}/{self.case_name}")
        save_processed_var_to_pickle(
            pairs_list, f'RP_{set_name}_pairs_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")
        save_processed_var_to_pickle(
            integer_labels, f'RP_{set_name}_labels_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")

class TrainSetConverter(DataSetConverter):
    """
    Extends DataSetConverter for converting ontology data into training and validation datasets specifically.
    Handles data loading, splitting, balancing, and generating new triples for training purposes.
    """
    def __init__(self, case_name):
        super().__init__("train", case_name) 
        self.triples = pd.DataFrame(columns=['head', 'relation', 'tail'])
        ontology_list = sorted(case_name.split(","))
        self.case_name = sort_words_str(case_name)
        for ontology in ontology_list:
            self.triples = pd.concat(
                [self.triples, read_processed_csv(f"{ontology}_triples", "triples")],
                ignore_index=True
            )


    def split_into_train_and_val(self):
        """
        Splits the dataset into training and validation sets based on a 20% split, maintaining proportionate distribution of relations.
        """
        total_count = len(self.triples)
        count_20_percent = total_count // 5

        relation_counts = self.triples['relation'].value_counts()
        relation_proportions = relation_counts / total_count

        validation_triples = pd.DataFrame(columns=self.triples.columns)
        train_triples = pd.DataFrame(columns=self.triples.columns)

        for relation, count in relation_counts.items():
            relation_triples = self.triples[self.triples['relation'] == relation]

            val_count = int(relation_proportions[relation] * count_20_percent)

            val_relation_triples = relation_triples.sample(n=val_count)
            train_relation_triples = relation_triples.drop(val_relation_triples.index)

            validation_triples = pd.concat([validation_triples, val_relation_triples])
            train_triples = pd.concat([train_triples, train_relation_triples])

        validation_triples = validation_triples.drop_duplicates(['head','tail'])

        self.triples = train_triples
        self.validation_triples = validation_triples

    def undersample_classes(self):
        """
        Balances the dataset by undersampling classes to match the least represented class's size.
        """
        self.triples = self.triples.iloc[::-1].reset_index(drop=True)
        count_min_class = self.triples['relation'].value_counts().min()
        dfs = []
        for label in self.triples['relation'].unique():
            df_class = self.triples[self.triples['relation'] == label]
            df_class_downsampled = resample(
                df_class, replace=False, n_samples=count_min_class, random_state=234
            )
            dfs.append(df_class_downsampled)
        df_balanced = pd.concat(dfs, ignore_index=True)
        self.triples = df_balanced
        self.balanced_bool = True

    def balance_relations(self) -> None:
        """
        Balances the number of 'exactmatch' triples with the higher count of 'broadmatch' or 'narrowmatch' triples, if necessary.
        """
        subsumption_mask = self.triples['relation'].isin(['broadmatch','narrowmatch'])
        subsumption_relation_counts = self.triples[subsumption_mask]['relation'].value_counts()
        max_count = np.max(subsumption_relation_counts)
        exactmatch_count = len(self.triples[self.triples['relation']=='exactmatch'])

        if max_count > exactmatch_count:
            taser_trainset = read_raw_csv("wordnet_triples")
            taser_trainset = taser_trainset.dropna()
            taser_trainset_equivalence = taser_trainset[taser_trainset['relation']=="exactmatch"]
            samples_equivalence = taser_trainset_equivalence.sample(n=(max_count-exactmatch_count+1))
            self.triples = pd.concat([self.triples, samples_equivalence])
        else:
            print("There are more exactmatch triples than subsumptions, so no extra exactmatches are added")    

    def shuffle_triples_grouped(self, batch_size=16):
        """
        Shuffles triples, ensuring that grouped triples (by head and tail) are shuffled within themselves and then batched.
        """
        grouped = self.triples.groupby(['head', 'tail'])
        group_keys = list(grouped.groups.keys())
        random.Random(3).shuffle(group_keys)
        batches = []
        current_batch = []

        for key in group_keys:
            group = grouped.get_group(key)
            indexes = group.index.tolist()
            random.Random(3).shuffle(indexes)
            current_batch.extend(indexes)

            if len(current_batch) >= batch_size:
                batches.append(current_batch[:batch_size])
                current_batch = current_batch[batch_size:]

        if current_batch:
            batches.append(current_batch)

        self.triples = pd.concat(
            [self.triples.loc[batches[i]] for i in range(len(batches))],
            ignore_index=True
        )


    def generate_selfloop_exactmatch_triples(self):
        """
        Generates self-loop 'exactmatch' triples to balance the count of 'exactmatch' triples with 'broadmatch' triples.
        """
        broadmatch_triples = self.triples[self.triples['relation']=="broadmatch"]
        number_broadmatches = len(broadmatch_triples)

        exactmatch_triples = self.triples[self.triples['relation']=="exactmatch"]
        number_exactmatches = len(exactmatch_triples)
        
        number_exactMatches_needed = number_broadmatches - number_exactmatches
        
        selfloop_exactmatches_head = broadmatch_triples.copy()
        selfloop_exactmatches_head['tail'] = selfloop_exactmatches_head['head']
        selfloop_exactmatches_head['relation'] = 'exactmatch'

        selfloop_exactmatches_tail = broadmatch_triples.copy()
        selfloop_exactmatches_tail['head'] = selfloop_exactmatches_tail['tail']
        selfloop_exactmatches_tail['relation'] = 'exactmatch'

        selfloop_exactmatches = pd.concat([selfloop_exactmatches_head, selfloop_exactmatches_tail])

        selfloop_exactmatches = selfloop_exactmatches.drop_duplicates()
        selfloop_exactmatches = selfloop_exactmatches.sample(frac=1.0)

        selfloop_selection = []
        for index, row in selfloop_exactmatches.iterrows():
            if not ((exactmatch_triples['head'] == row['head']) & (exactmatch_triples['tail'] == row['tail'])).any():
                selfloop_selection.append(row)
                number_exactMatches_needed -= 1
            if number_exactMatches_needed <= 0:
                break

        all_triples = pd.concat([self.triples, pd.DataFrame(selfloop_selection)], ignore_index=True)

        self.triples = all_triples

    def print_triples_statistics(self) -> None:
        """
        Prints statistics of the triples, including total number and distribution of relations and labels.
        """
        print("-------------- Triple statistics ------------------")
        print(f'Number of triples: {len(self.triples)}')

        relation_counts = self.triples['relation'].value_counts().to_dict()
        print("Relation Counts Top 20:")
        counter = 0
        for relation, count in relation_counts.items():
            print(f"{relation}: {count}")
            if counter >= 19:
                break
            counter += 1

        if "label" in self.triples.columns:
            label_counts = self.triples['label'].value_counts().to_dict()
            print("Label Counts:")
            for label, count in label_counts.items():
                print(f"{label}: {count}")


class TestSetConverter(DataSetConverter):
    """
    Extends DataSetConverter to specifically handle the conversion of test set data into formats suitable for evaluation and testing purposes. This class focuses on test sets for scenarios such as STROMA project evaluations.
    """
    def __init__(self, case_name):
        super().__init__("test", case_name) 
        self.test_triples = read_processed_csv(f"{case_name}_triples", "triples")
        self.balanced_bool = None

    def convert_triples_to_stroma_format(self) -> None:
        """
        Converts the test set triples into a specific format required by the STROMA project, saving the formatted data to a text file.
        """
        only_pair = self.test_triples[['head','tail']]
        list_of_strings = only_pair.apply(lambda x: ' :: '.join(x.astype(str)), axis=1).tolist()

        with open(f'../data/processed/test/default/{self.case_name}/STROMA_test_{self.case_name}.txt', 'w') as file:
            for line in list_of_strings:
                file.write(line + '\n')


class AnchorSetsCreator(DataSetConverter):
    """
    Creates anchor sets from training and testing data for Relation Prediction (RP) tasks,
        supporting the creation of evaluation sets that include a specified percentage of anchor points from the test set.
    """
    def __init__(self, case_name, anchor_percentage):
        """
        Initializes the AnchorSetsCreator with a case name and an anchor percentage, loading the appropriate training and test triples.
        """
        super().__init__(None, case_name) 
        self.balanced_bool = False
        self.triples = read_processed_csv(f"RP_train_{case_name}_UB", f"train/default/{case_name}")

        test_case_name_dict = {"stroma_g2_source_stroma_g2_target":"stroma_g2_reference",
                        "stroma_g4_source_stroma_g4_target":"stroma_g4_reference",
                        "stroma_g5_source_stroma_g5_target":"stroma_g5_reference",
                        "stroma_g6_source_stroma_g6_target":"stroma_g6_reference",
                        "stroma_g7_source_stroma_g7_target":"stroma_g7_reference",
                        "CNL_ESCO":"ESCO-CNL",
                        "ESCO_ONET":"ESCO-ONET"}
        self.test_case_name = test_case_name_dict[case_name]

        self.test_triples = read_processed_csv(f"RP_test_{self.test_case_name}", f"test/default/{self.test_case_name}")
        self.anchor_percentage = anchor_percentage

    def generate_train_test_sets(self):
        """
        Generates training and testing sets by selecting a subset of the test set as anchors based on the specified anchor percentage, then recombining these anchors with the training set.
        """
        test_triples = self.test_triples.copy()

        total_anchors = int(len(self.test_triples)*self.anchor_percentage)
        anchors = test_triples.sample(n=total_anchors)

        print(len(self.test_triples))
        print(total_anchors)

        self.test_triples = self.test_triples.drop(anchors.index)

        print(len(self.test_triples))
        print(self.test_triples['relation'].value_counts())
        print(len(anchors))
        print(anchors['relation'].value_counts())

        self.triples = pd.concat([self.triples, anchors])

    def convert_triples_to_stroma_format(self) -> None:
        """
        Converts the test triples to a format suitable for use with STROMA, specifically converting triples to a pair string format and saving to a file.
        """
        only_pair = self.test_triples[['head','tail']]
        list_of_strings = only_pair.apply(lambda x: ' :: '.join(x.astype(str)), axis=1).tolist()

        with open(f'../data/processed/test/anchors_{int(self.anchor_percentage*100)}/{self.test_case_name}/STROMA_test_{self.test_case_name}.txt', 'w') as file:
            for line in list_of_strings:
                file.write(line + '\n')

def main(args):
    if args.action == "CNL_ontology_to_triples":
        run_cnl_ontology()
    elif args.action == "ESCO_ontology_to_triples":
        run_esco_ontology()
    elif args.action == "ONET_ontology_to_triples":
        run_onet_ontology()
    elif args.action == "bio_ontologies_to_triples":
        run_bio_ontologies()
    elif args.action == "bio_mappings_to_triples":
        run_bio_mappings()
    elif args.action == "ESCO-CNL_mapping_to_triples":
        run_esco_cnl_mapping()
    elif args.action == "ESCO-ONET_mapping_to_triples":
        run_esco_onet_mapping()
    elif args.action == 'ESCO-BLM_mapping_to_triples':
        run_esco_blm_mapping()
    elif args.action == "stroma_ontologies_to_triples":
        run_stromas_ontologies()
    elif args.action == "stroma_mappings_to_triples":
        run_stromas_mapping()
    elif args.action == "stroma_external_data_sets_to_triples":
        run_stroma_external()
    elif args.action == 'train_set_converter':
        run_train_set_converter(args)
    elif args.action == "test_set_converter":
        run_test_set_converter(args)
    elif args.action == "anchor_creator":
        run_anchor_creator(args)
    else:
        raise ValueError(f"Unknown case: {args.action}. Try again.")

def run_cnl_ontology():
    preprocessor = CNLOntologyPreprocessor("CNL")
    preprocessor.preprocess_raw_ontology()
    preprocessor.convert_to_graph()
    preprocessor.generate_subsumption_triples()
    preprocessor.remove_plural_subsumptions()
    preprocessor.remove_duplicate_pairs()
    preprocessor.save_triples()

def run_esco_ontology():
    preprocessor = ESCOOntologyPreprocessor("ESCO")
    preprocessor.convert_to_graph()
    preprocessor.generate_subsumption_triples()
    preprocessor.generate_exactmatch_ESCO()
    preprocessor.remove_duplicate_pairs()
    preprocessor.save_triples()

def run_onet_ontology():
    preprocessor = ONETOntologyPreprocessor("ONET")
    preprocessor.read_SOC()
    preprocessor.read_ONET()
    preprocessor.convert_to_graph()
    preprocessor.generate_subsumption_triples()
    preprocessor.remove_duplicate_pairs()
    preprocessor.fix_inconsistencies()
    preprocessor.save_triples()

def run_bio_ontologies():
    for case in ['ncit-doid', 'omim-ordo', 'snomed.body-fma.body', 'snomed.neoplas-ncit.neoplas', 'snomed.pharm-ncit.pharm']:
        for ontology in case.split('-'):
            preprocessor = BIOOntologyPreprocessor(ontology)
            preprocessor.subset_exactmatches()
            preprocessor.generate_subsumption_triples_bio()
            preprocessor.remove_duplicate_pairs()
            preprocessor.save_triples()

def run_bio_mappings():
    for case in ['ncit-doid', 'omim-ordo', 'snomed.body-fma.body', 'snomed.neoplas-ncit.neoplas', 'snomed.pharm-ncit.pharm']:
        preprocessor = BioMappingPreprocessor(case)
        preprocessor.process_raw_mapping()
        preprocessor.replace_urls_with_labels()
        preprocessor.generate_narrowmatch_triples()
        preprocessor.subsample_if_necessary()
        preprocessor.save_mapping_triples()

def run_esco_cnl_mapping():
    preprocessor = ESCOCNLMappingPreprocessor()
    run_common_mapping_steps(preprocessor)
    preprocessor.remove_closematch_rows()
    preprocessor.save_mapping_triples()

def run_esco_onet_mapping():
    preprocessor = ESCOONETMappingPreprocessor()
    run_common_mapping_steps(preprocessor)
    preprocessor.save_mapping_triples()

def run_esco_blm_mapping():
    preprocessor = ESCOBLMMappingPreprocessor()
    run_common_mapping_steps(preprocessor)
    preprocessor.remove_closematch_rows()
    preprocessor.save_mapping_triples()

def run_common_mapping_steps(preprocessor):
    preprocessor.process_raw_mapping()
    preprocessor.drop_duplicate_rows()

def run_stromas_ontologies():
    dataset_ids = ['g2', 'g4', 'g5', 'g6']
    files = ['source','target']
    for dataset_id in dataset_ids:
        for file in files:
            case_name = dataset_id + "_" + file
            preprocessor = STROMAOntologyPreprocessor(case_name)
            preprocessor.RDF_to_df()
            preprocessor.get_inverse_triples()
            preprocessor.save_triples()

def run_stromas_mapping():
    dataset_ids = ['g2', 'g4', 'g5', 'g6', 'g7']
    for dataset_id in dataset_ids:
        preprocessor = RDFMappingPreprocessor(dataset_id)
        preprocessor.RDF_to_df()
        if dataset_id == 'g6':
            preprocessor.generate_broadmatch_triples()
        preprocessor.drop_duplicate_rows()
        preprocessor.save_mapping_triples()

def run_stroma_external():
    dataset_names = ['dbpedia', 'schema_org', 'wordnet', 'wikidata']
    for dataset_name in dataset_names:
        preprocessor = StromaExternalDataPreprocessor(dataset_name)
        preprocessor.convert_ints_to_labels()
        preprocessor.save_triples()

def run_train_set_converter(args):
    check_converter_args(args)
    preprocessor = TrainSetConverter(args.data_sets)
    preprocessor.strip_rows("train")
    preprocessor.print_triples_statistics()
    preprocessor.drop_rows_with_na("train")
    preprocessor.print_triples_statistics()
    preprocessor.generate_selfloop_exactmatch_triples()
    preprocessor.print_triples_statistics()
    preprocessor.drop_duplicate_rows("train")
    preprocessor.print_triples_statistics()

    if args.task_type == "RP":
        if args.balanced == "T":
            preprocessor.undersample_classes()
        preprocessor.save_to_pairs_RP("train", "default")

    elif args.task_type == "TC":
        if args.balanced == "T":
            preprocessor.undersample_classes()
        preprocessor.add_pos_label_to_df("train")
        preprocessor.generate_all_negatives("train")
        preprocessor.shuffle_triples_grouped()
        preprocessor.save_to_triples_TC("train", "default")
    else:
        sys.exit(f"Unknown task type: {args.task_type}. Try again")

    preprocessor.print_triples_statistics()

def run_test_set_converter(args):
    check_converter_args(args)
    preprocessor = TestSetConverter(args.data_sets)
    preprocessor.strip_rows("test")
    preprocessor.drop_rows_with_na("test")
    preprocessor.drop_duplicate_rows("test")
    if args.task_type == "RP":
        preprocessor.save_to_pairs_RP("test", "default")
        preprocessor.convert_triples_to_stroma_format()
    elif args.task_type == "TC":
        preprocessor.add_pos_label_to_df("test")
        preprocessor.generate_all_negatives("test")
        preprocessor.save_to_triples_TC("test", "default")
    else:
        sys.exit(f"Unknown task type: {args.task_type}. Try again")

def run_anchor_creator(args):
    percentages = [0.80]
    set_names = ['train','test']
    for percentage in percentages:
        preprocessor = AnchorSetsCreator(args.data_sets, percentage)
        preprocessor.generate_train_test_sets()
        preprocessor.drop_duplicate_rows("train")
        preprocessor.shuffle_triples_random()

        for set_name in set_names:
            readable_percentage = "anchors_" + str(int(percentage*100))
            preprocessor.save_to_pairs_RP(set_name, readable_percentage)
            preprocessor.convert_triples_to_stroma_format()
            preprocessor.add_pos_label_to_df(set_name)
            preprocessor.generate_all_negatives(set_name)
            preprocessor.save_to_triples_TC(set_name, readable_percentage)

def check_converter_args(args):
    if args.action == "train_set_converter":    
        if (args.data_sets is None) or (args.task_type is None) or (args.task_type == "RP" and args.balanced is None):
            sys.exit("Missing arguments for converter, check if you need to add data_sets, task_type or balanced")
    elif args.action == "test_set_converter":
        if (args.data_sets is None) or (args.task_type is None):
            sys.exit("Missing arguments for converter, make sure you have specified data_sets and task_type")
    elif (args.action == "train_set_converter_matcher") or (args.action == "test_set_converter_matcher"):
        if (args.data_sets is None):
            sys.exit("Missing arguments for matcher converter, make sure you have specified data_sets")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run data preparations, there are various options.")
    parser.add_argument("--action", required=True, type=str, help="action to be completed")
    parser.add_argument("--data_sets", required=False, type=str, help="datasets to be converted")
    parser.add_argument("--task_type", required=False, type=str, help="choose the task type")
    parser.add_argument("--balanced", required=False, type=str, help="choose whether to balance the classes")
    args = parser.parse_args()
    main(args)