from xml.etree.ElementTree import ElementTree
import pandas as pd
from tqdm import tqdm

from global_parameters import LABELS_MAP


def xml_to_df(path: str, xml_file_names: list[str]):

    data_sets = [[], []]

    tree = ElementTree()

    hum_xml = tree.parse(path + xml_file_names[0])
    hum_records = hum_xml.findall('.//Rec')

    vet_xml = tree.parse(path + xml_file_names[1])
    vet_records = vet_xml.findall('.//Rec')

    record_sets = [hum_records, vet_records]

    progress_bar = tqdm(range(len(hum_records + vet_records)))

    for i, med_field in enumerate(LABELS_MAP):
        print(f"Processing {xml_file_names[i]}")
        labels = [1, 0] if med_field == "human" else [0, 1]
        for rec in record_sets[i]:
            try:
                common = rec.find('.//Common')
                pmid = common.find('PMID').text
                text_types = [elem.text for elem in common.findall('Type')]
                title = common.find('Title').text
                abstract = common.find('Abstract').text
                mesh_term_list = rec.find('.//MeshTermList')
                mesh_terms = [
                    term.text for term in mesh_term_list.findall('MeshTerm')]
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Error occured for PMID: {pmid}")

            data_sets[i].append({'pmid': pmid, "text_types": text_types, 'title': title,
                                 'abstract': abstract, 'meshtermlist': mesh_terms, 'labels': labels})
            progress_bar.update(1)

    hum_df = pd.DataFrame(data_sets[0])
    vet_df = pd.DataFrame(data_sets[1])

    return hum_df, vet_df
